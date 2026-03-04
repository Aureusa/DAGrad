from abc import ABC
from collections import deque
import inspect
from typing import Optional

from .block import Block


class Workflow(ABC):
    INPUT_NODE = "__workflow_input__"

    def __init__(self):
        """
        Initializes a workflow with a name and an empty list of blocks.

        :param name: A unique identifier for the workflow.
        :type name: str
        """
        self.name = self.__class__.__name__
        self.blocks = []
        self._blocks_by_key: dict[str, Block] = {}
        self._edges: list[tuple[str, str, str, str]] = []
        self._input_edges: list[tuple[str, str, str]] = []
        self._workflow_outputs: dict[str, tuple[str, str]] = {}

        self._validated_graph = False

    def add_block(self, block: Block, key: Optional[str] = None):
        """
        Adds a block to the workflow.

        If no explicit graph connections are added, blocks execute sequentially in insertion order.
        For graph mode, use ``key`` to assign a stable node identifier and then connect blocks
        with ``connect`` / ``connect_input``.

        :param block: The block to be added to the workflow.
        :type block: Block
        :param key: Optional unique identifier for the block in graph mode. Defaults to block class name.
        :type key: str, optional
        :raises ValueError: If the block is not an instance of Block.
        """
        if not isinstance(block, Block):
            raise ValueError(f"All blocks must be of type Block, got {type(block)}")

        node_key = key or block.name
        if node_key in self._blocks_by_key:
            raise ValueError(f"Duplicate block key '{node_key}'. Keys must be unique.")

        self.blocks.append(block)
        self._blocks_by_key[node_key] = block
        return node_key

    def connect(self, src: str, dst: str, src_output: str = "out", dst_input: str = "x"):
        """
        Connects one block output to another block input.

        :param src: Source block key.
        :type src: str
        :param dst: Destination block key.
        :type dst: str
        :param src_output: Name of source output port. Defaults to "out".
        :type src_output: str
        :param dst_input: Name of destination input port. Defaults to "x".
        :type dst_input: str
        """
        self._edges.append((src, src_output, dst, dst_input))
        return self

    def connect_input(self, input_name: str, dst: str, dst_input: str = "x"):
        """
        Connects a workflow-level input to a block input.

        :param input_name: Workflow input name expected by ``run``.
        :type input_name: str
        :param dst: Destination block key.
        :type dst: str
        :param dst_input: Name of destination input port. Defaults to "x".
        :type dst_input: str
        """
        self._input_edges.append((input_name, dst, dst_input))
        return self

    def set_outputs(self, outputs: dict[str, tuple[str, str]]):
        """
        Defines named workflow outputs.

        :param outputs: Mapping of workflow output name to (block_key, output_port).
        :type outputs: dict[str, tuple[str, str]]
        """
        self._workflow_outputs = outputs
        return self

    def _uses_graph_mode(self):
        return bool(self._edges or self._input_edges or self._workflow_outputs)

    def _validate_graph(self):
        if not self._blocks_by_key and self._uses_graph_mode():
            raise ValueError("Graph mode requires at least one block.")

        node_keys = set(self._blocks_by_key.keys())
        incoming_ports: dict[str, set[str]] = {key: set() for key in node_keys}

        for src, src_output, dst, dst_input in self._edges:
            if src not in node_keys:
                raise ValueError(f"Unknown source block '{src}' in connect().")
            if dst not in node_keys:
                raise ValueError(f"Unknown destination block '{dst}' in connect().")
            if dst_input in incoming_ports[dst]:
                raise ValueError(
                    f"Input port '{dst_input}' of block '{dst}' is connected more than once."
                )
            incoming_ports[dst].add(dst_input)

            if not isinstance(src_output, str) or not src_output:
                raise ValueError("Source output port must be a non-empty string.")

        for input_name, dst, dst_input in self._input_edges:
            if dst not in node_keys:
                raise ValueError(f"Unknown destination block '{dst}' in connect_input().")
            if dst_input in incoming_ports[dst]:
                raise ValueError(
                    f"Input port '{dst_input}' of block '{dst}' is connected more than once."
                )
            incoming_ports[dst].add(dst_input)

            if not isinstance(input_name, str) or not input_name:
                raise ValueError("Workflow input name must be a non-empty string.")

        for out_name, (node, port) in self._workflow_outputs.items():
            if node not in node_keys:
                raise ValueError(
                    f"Workflow output '{out_name}' references unknown block '{node}'."
                )
            if not isinstance(port, str) or not port:
                raise ValueError(
                    f"Workflow output '{out_name}' must reference a non-empty output port."
                )

        self._topological_order()

    def _topological_order(self):
        node_keys = list(self._blocks_by_key.keys())
        indegree = {k: 0 for k in node_keys}
        adjacency = {k: [] for k in node_keys}

        for src, _, dst, _ in self._edges:
            indegree[dst] += 1
            adjacency[src].append(dst)

        queue = deque([k for k in node_keys if indegree[k] == 0])
        order = []

        while queue:
            node = queue.popleft()
            order.append(node)
            for nxt in adjacency[node]:
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    queue.append(nxt)

        if len(order) != len(node_keys):
            raise ValueError("Workflow graph contains a cycle.")

        return order

    @staticmethod
    def _normalize_outputs(result):
        if isinstance(result, dict):
            normalized = {}
            for key, value in result.items():
                if not isinstance(key, str):
                    raise ValueError("Block output dictionary keys must be strings.")
                normalized[key] = value
            return normalized
        return {"out": result}

    @staticmethod
    def _execute_block(block: Block, inputs: dict):
        if not inputs:
            return block.execute()

        signature = inspect.signature(block.execute)
        params = list(signature.parameters.values())
        has_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)
        accepted_kwargs = {
            p.name
            for p in params
            if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }

        if has_var_kwargs or set(inputs.keys()).issubset(accepted_kwargs):
            return block.execute(**inputs)

        if len(inputs) == 1:
            return block.execute(next(iter(inputs.values())))

        raise ValueError(
            f"Block '{block.name}' cannot accept graph inputs {list(inputs.keys())}. "
            "Use execute(self, **inputs) or matching named arguments."
        )

    def run(self, input_data):
        """
        Executes the workflow.

        - Sequential mode (default): if no graph connections are defined, runs blocks in insertion order.
        - Graph mode: if ``connect`` / ``connect_input`` / ``set_outputs`` were used, executes blocks
          in topological order.

        :param input_data: Workflow input data. In graph mode, use a dict for multiple named inputs.
        :type input_data: Any
        :return: Sequential output in sequential mode; graph outputs in graph mode.
        :rtype: Any
        """
        if not self._uses_graph_mode():
            out = input_data
            for block in self.blocks:
                out = block.execute(out)
            return out

        if not self._validated_graph:
            self._validate_graph()
            self._validated_graph = True

        incoming: dict[str, dict[str, tuple[str, str]]] = {
            key: {} for key in self._blocks_by_key.keys()
        }
        outgoing_count = {key: 0 for key in self._blocks_by_key.keys()}

        for src, src_output, dst, dst_input in self._edges:
            incoming[dst][dst_input] = (src, src_output)
            outgoing_count[src] += 1

        for input_name, dst, dst_input in self._input_edges:
            incoming[dst][dst_input] = (self.INPUT_NODE, input_name)

        if isinstance(input_data, dict):
            workflow_inputs = input_data
        else:
            expected_input_names = {name for name, _, _ in self._input_edges}
            if len(expected_input_names) > 1:
                raise ValueError(
                    "Graph workflow expects multiple named inputs. "
                    "Pass a dict to run(input_data)."
                )
            if len(expected_input_names) == 1:
                workflow_inputs = {next(iter(expected_input_names)): input_data}
            else:
                workflow_inputs = {"input": input_data}

        execution_cache: dict[str, dict[str, object]] = {}
        for node_key in self._topological_order():
            node_inputs = {}
            for dst_input, (src_node, src_port) in incoming[node_key].items():
                if src_node == self.INPUT_NODE:
                    if src_port not in workflow_inputs:
                        raise ValueError(
                            f"Missing workflow input '{src_port}' required by block '{node_key}'."
                        )
                    node_inputs[dst_input] = workflow_inputs[src_port]
                    continue

                if src_port not in execution_cache[src_node]:
                    raise ValueError(
                        f"Block '{src_node}' did not produce output port '{src_port}'."
                    )
                node_inputs[dst_input] = execution_cache[src_node][src_port]

            block = self._blocks_by_key[node_key]
            result = self._execute_block(block, node_inputs)
            execution_cache[node_key] = self._normalize_outputs(result)

        if self._workflow_outputs:
            outputs = {}
            for out_name, (node_key, port) in self._workflow_outputs.items():
                if port not in execution_cache[node_key]:
                    raise ValueError(
                        f"Workflow output '{out_name}' references missing port '{port}' "
                        f"from block '{node_key}'."
                    )
                outputs[out_name] = execution_cache[node_key][port]
            return outputs

        sink_nodes = [node for node, count in outgoing_count.items() if count == 0]
        if len(sink_nodes) == 1:
            sink_outputs = execution_cache[sink_nodes[0]]
            if set(sink_outputs.keys()) == {"out"}:
                return sink_outputs["out"]
            return sink_outputs

        return {node: execution_cache[node] for node in sink_nodes}

    def summary(self):
        """
        Returns a compact summary of graph metadata.

        :return: Dictionary with graph execution metadata.
        :rtype: dict
        """
        if not self._uses_graph_mode():
            return {
                "mode": "sequential",
                "blocks": [b.name for b in self.blocks],
            }

        self._validate_graph()
        return {
            "mode": "graph",
            "nodes": list(self._blocks_by_key.keys()),
            "edges": [
                {
                    "src": src,
                    "src_output": src_output,
                    "dst": dst,
                    "dst_input": dst_input,
                }
                for src, src_output, dst, dst_input in self._edges
            ],
            "input_edges": [
                {
                    "input": input_name,
                    "dst": dst,
                    "dst_input": dst_input,
                }
                for input_name, dst, dst_input in self._input_edges
            ],
            "outputs": self._workflow_outputs,
            "topological_order": self._topological_order(),
        }

    def parameters(self):
        """
        Yields the parameters of all blocks in the workflow.

        :return: An iterator over the parameters of all blocks.
        :rtype: Iterator
        """
        for block in self.blocks:
            yield from block.parameters()

    def to(self, device):
        """Moves all blocks in the workflow to the specified device."""
        for block in self.blocks:
            block.to(device)
        return self

    def __repr__(self):
        if not self._uses_graph_mode():
            return f"Workflow(name={self.name}, blocks={[b.name for b in self.blocks]})"
        return (
            f"Workflow(name={self.name}, "
            f"nodes={list(self._blocks_by_key.keys())}, "
            f"edges={len(self._edges)}, "
            f"input_edges={len(self._input_edges)})"
        )

    def __str__(self):
        if not self.blocks:
            return f"{self.name}: <no blocks>"

        if self._uses_graph_mode():
            incoming_by_dst: dict[str, list[tuple[str, str, str]]] = {
                key: [] for key in self._blocks_by_key.keys()
            }
            for src, src_output, dst, dst_input in self._edges:
                if dst in incoming_by_dst:
                    incoming_by_dst[dst].append((src, src_output, dst_input))

            for input_name, dst, dst_input in self._input_edges:
                if dst in incoming_by_dst:
                    incoming_by_dst[dst].append((self.INPUT_NODE, input_name, dst_input))

            try:
                ordered_nodes = self._topological_order()
            except ValueError:
                ordered_nodes = list(self._blocks_by_key.keys())

            lines = [f"{self.name}: [graph mode]"]
            lines.append("")

            reported_start_of_graph = False
            for index, node_key in enumerate(ordered_nodes):
                block = self._blocks_by_key[node_key]
                # lines.append(f"[{node_key}] {block.name}")

                incoming_edges = incoming_by_dst.get(node_key, [])
                if incoming_edges:
                    for src, src_output_or_input, dst_input in incoming_edges:
                        if src == self.INPUT_NODE:
                            if index == 0:
                                lines.append(
                                    f"  Workflow Input:"
                                )
                                reported_start_of_graph = True
                            lines.append(
                                f"      [Workflow Input] OutputPort({src_output_or_input}) ──▶ [{block.name}] InputPort({dst_input})"
                            )
                        else:
                            if reported_start_of_graph:
                                lines.append("  Past workflow input nodes:")
                                reported_start_of_graph = False
                            lines.append(
                                f"      [{self._blocks_by_key[src].name}] OutputPort({src_output_or_input}) ──▶ [{block.name}] InputPort({dst_input})"
                            )
                else:
                    lines.append(f"       <no incoming connections> ──▶ [{block.name}] InputPort()")

                # block_lines = str(block).split("\n")
                # lines.extend(f"  {line}" for line in block_lines)

                if index != len(ordered_nodes) - 1:
                    lines.append("")

            if self._workflow_outputs:
                lines.append("")
                lines.append("  Workflow outputs:")
                for out_name, (node, port) in self._workflow_outputs.items():
                    lines.append(f"     [{self._blocks_by_key[node].name}] OutputPort({port}) ──▶ {out_name}")

            return "\n".join(lines)

        lines = [f"{self.name}:"]
        for i, block in enumerate(self.blocks):
            lines.append(f"{block}")
            if i != len(self.blocks) - 1:
                lines.append("   │")
                lines.append("   ▼")
        return "\n".join(lines)



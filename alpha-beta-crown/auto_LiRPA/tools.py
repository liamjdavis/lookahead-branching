import torch
from graphviz import Digraph
import shutil
import re

from typing import TYPE_CHECKING, List
if TYPE_CHECKING:
    from .bound_general import BoundedModule


def visualize(self: 'BoundedModule', output_path):
    r"""A visualization tool for BoundedModule.
    If dot engine is available in the system enviornment, it renders the graph and output {output_path}.png.
    Otherwise, it output a {output_path}.dot.
    """

    nodes = list(self.nodes())
    # Create a directed graph
    dot = Digraph(format='png', engine='dot')
    # Add nodes with optional attributes
    for node in nodes:
        # we name the Graphviz nodes with the sanitized node name,
        # while keeping the original name in the label which is displayed in the graph.
        export_node_name = sanitize_graphviz_name(node.name)
        label = f"""<
            <TABLE BORDER="0" CELLBORDER="0" CELLPADDING="4">
                <TR><TD><FONT FACE="Arial" COLOR="black">{node.name}</FONT></TD></TR>
                <TR><TD><FONT FACE="Courier" COLOR="blue">{node.__class__.__name__}</FONT></TD></TR>
                <TR><TD><FONT FACE="Courier" COLOR="black">{
                    tuple(node.output_shape) if node.output_shape is not None else None}</FONT></TD></TR>
            </TABLE>
        >"""
        # perturbed nodes are highlighted in grey
        if getattr(node, "perturbed", False):
            style_attrs = {'style': 'filled', 'fillcolor': 'lightgrey'}
        else:
            style_attrs = {}
        if node.__class__.__name__ in ["BoundParams", "boundConstant", "BoundBuffers"]:
            dot.node(export_node_name, label=label, fontsize="8", width="0.5", height="0.2", shape="ellipse", **style_attrs)
        elif node.__class__.__name__ == "BoundInput":
            dot.node(export_node_name, label=label, shape="diamond", **style_attrs)
        else:
            dot.node(export_node_name, label=label, shape="square", **style_attrs)
        for inp in node.inputs:
            dot.edge(sanitize_graphviz_name(inp.name), export_node_name)
    # Render graph
    if shutil.which("dot") is None:
        print("Cannot render the graphviz file (dot not found).")
        print(f"Graph saved to {output_path}.dot")
        dot.save(output_path + ".dot")
    else:
        dot.render(output_path, cleanup=True)
        print(f"Graph saved to {output_path}.png")

def sanitize_graphviz_name(name):
    """
    Convert problematic characters (like `:`, `::`) in a Graphviz node name to a safe alternative character `_`.
    """
    unsafe_chars = r'[:;,\[\]{}()<>|#*@&=+`~^?"\\\s]'
    safe_name = re.sub(unsafe_chars, "_", name)
    
    return safe_name

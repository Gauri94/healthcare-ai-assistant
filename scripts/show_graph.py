"""Print the LangGraph agent as a Mermaid diagram.

Run:   uv run python -m scripts.show_graph
Then paste the output into https://mermaid.live to see it drawn, or into a
Markdown file: GitHub renders ```mermaid code blocks as diagrams.

CONCEPT: LangGraph knows the graph's structure (nodes, edges, and where each
conditional edge can lead, from the routers' Literal[...] return types), so it
can draw itself. Handy for docs, debugging, and explaining the design in interviews.
"""

from app.graph_agent import build_graph

print(build_graph().get_graph().draw_mermaid())

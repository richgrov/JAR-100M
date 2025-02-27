from pygls.server import LanguageServer
from lsprotocol import types

server = LanguageServer("jar-language-server", "0.1")

@server.feature(types.TEXT_DOCUMENT_COMPLETION)
def completion(params: types.CompletionParams):
    items = []
    document = server.workspace.get_text_document(params.text_document.uri)
    current_line = document.lines[params.position.line].strip()

    if (current_line.endswith("hello.")):
        items = [
            types.CompletionItem(label="hello"),
            types.CompletionItem(label="world")
        ]

    return types.CompletionList(is_incomplete=false, items=items)

server.start_io()
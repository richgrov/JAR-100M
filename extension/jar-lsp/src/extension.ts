import * as vscode from 'vscode';

export function activate(context: vscode.ExtensionContext) {
	console.log('Congratulations, your extension "jar-lsp" is now active!');

	const disposable = vscode.commands.registerCommand('jar-lsp.helloWorld', () => {
		vscode.window.showInformationMessage('Hello World from JAR-LSP!')
	});

	context.subscriptions.push(disposable);
}

export function deactivate() {}

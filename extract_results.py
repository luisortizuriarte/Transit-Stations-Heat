import json
import glob

with open('notebook_results.txt', 'w', encoding='utf-8') as out:
    for nb in glob.glob('*.ipynb'):
        out.write(f'\n\n====== Notebook: {nb} ======\n')
        try:
            with open(nb, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for cell in data.get('cells', []):
                # We can also extract markdown cells that might contain result interpretations
                if cell.get('cell_type') == 'markdown':
                    source = ''.join(cell.get('source', []))
                    if 'result' in source.lower() or 'conclusion' in source.lower() or 'figure' in source.lower():
                        out.write('Markdown discussing results: ' + source.strip()[:500].replace('\n', ' ') + '...\n')
                elif cell.get('cell_type') == 'code':
                    outputs = cell.get('outputs', [])
                    for output in outputs:
                        if output.get('output_type') == 'stream':
                            text = ''.join(output.get('text', []))
                            if text.strip():
                                out.write(f'Stream output: {text.strip()[:1000]}\n')
                        elif output.get('output_type') in ['execute_result', 'display_data']:
                            data_dict = output.get('data', {})
                            if 'text/plain' in data_dict:
                                text = ''.join(data_dict['text/plain'])
                                out.write(f'Text output: {text.strip()[:1000]}\n')
                        elif output.get('output_type') == 'error':
                            err = output.get('ename', '') + ': ' + output.get('evalue', '')
                            out.write(f'Error output: {err}\n')
        except Exception as e:
            out.write(f'Error reading {nb}: {e}\n')

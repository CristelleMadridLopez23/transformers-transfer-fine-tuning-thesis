"""Descarga las revisiones seleccionadas y verifica la integridad publicada."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
CLOUD_REVISION = '7cf997311dd19fcb3fde0585d22557656258197d'
CLOUD_SOURCE_REVISION = '9da4e7c7a5f507df2640315ad8085ad7bfaab3b5'

def get_json(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)

def download(url, path, size=None, expected=None, algorithm='sha256'):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.part')
    sha, md5 = hashlib.sha256(), hashlib.md5()
    count = 0
    with urllib.request.urlopen(url, timeout=120) as response, tmp.open('wb') as out:
        while chunk := response.read(1024 * 1024):
            out.write(chunk)
            sha.update(chunk)
            md5.update(chunk)
            count += len(chunk)
    if size is not None and count != size:
        raise ValueError(f'Tamaño incorrecto: {path}')
    actual = sha.hexdigest() if algorithm == 'sha256' else md5.hexdigest()
    if expected and actual != expected:
        raise ValueError(f'Hash incorrecto: {path}')
    tmp.replace(path)
    print(f'Descargado {path.relative_to(ROOT)}: {count} bytes', flush=True)
    return dict(path=str(path.relative_to(ROOT)), url=url, bytes=count,
                sha256=sha.hexdigest(), md5=md5.hexdigest(),
                upstream_hash=expected, upstream_hash_algorithm=algorithm)

def mattext():
    repo, rev = 'n0w0f/MatText-slices-2m', 'c21d7e29ccc7759b40fe2f3f0fe32a65f51b734e'
    records = []
    for f in get_json(f'https://huggingface.co/api/models/{repo}/tree/{rev}'):
        if f['type'] == 'file':
            records.append(download(f'https://huggingface.co/{repo}/resolve/{rev}/{f["path"]}',
                                    ROOT / 'mattext-slices-2m' / f['path'], f['size'],
                                    f.get('lfs', {}).get('oid')))
    return dict(repository=repo, revision=rev, files=records)

def cloud():
    repo, rev = 'ChangwenXu/CLOUD', CLOUD_REVISION
    # The model-info endpoint is accessible when the revision route is throttled.
    # Accept its metadata only if it still points to the pinned revision.
    metadata = get_json(f'https://huggingface.co/api/models/{repo}?blobs=true')
    if metadata['sha'] != rev:
        raise ValueError('CLOUD HEAD changed; pinned file metadata must be reviewed')
    records = []
    for f in metadata['siblings']:
        path = f['rfilename']
        records.append(download(f'https://huggingface.co/{repo}/resolve/{rev}/{path}',
                                ROOT / 'cloud' / path, f['size'],
                                f.get('lfs', {}).get('sha256')))
    source_repo, source_rev = 'BattModels/CLOUD', CLOUD_SOURCE_REVISION
    tree = get_json(f'https://api.github.com/repos/{source_repo}/git/trees/{source_rev}?recursive=1')
    source_records = []
    for f in tree['tree']:
        path = f['path']
        # Include conversion tables and code; omit benchmark/pretraining datasets.
        if f['type'] != 'blob' or path.startswith('example_data/'):
            continue
        if path.startswith('data/') and path not in {
            'data/generator.json', 'data/wyckoff-position-multiplicities.json'
        }:
            continue
        source_records.append(download(
            f'https://raw.githubusercontent.com/{source_repo}/{source_rev}/{path}',
            ROOT / 'cloud' / 'source' / path, f['size']))
    return dict(repository=repo, revision=rev, representation='SCOPE',
                checked_on='2026-10-20', files=records,
                source_repository=source_repo, source_revision=source_rev,
                source_files=source_records,
                excluded_datasets=['data/mb_cloud.json', 'example_data/'])

def matgpt():
    rev = 'a2eaee254bc829f0cc19edce5cbf27bbc3d35d5e'
    records = []
    tree = get_json(f'https://api.github.com/repos/devalab/matgpt/git/trees/{rev}?recursive=1')
    for f in tree['tree']:
        if f['type'] == 'blob':
            records.append(download(f'https://raw.githubusercontent.com/devalab/matgpt/{rev}/{f["path"]}',
                                    ROOT / 'matgpt' / 'source' / f['path'], f['size']))
    article = get_json('https://api.figshare.com/v2/articles/34330655')
    (ROOT / 'matgpt' / 'figshare-metadata.json').write_text(json.dumps(article, indent=2) + '\n')
    selected = {'mat-uncond.zip': 'unconditional', 'mat-multi.zip': 'formation-bandgap'}
    for f in article['files']:
        if f['name'] not in selected:
            continue
        archive = ROOT / 'matgpt' / 'archives' / f['name']
        records.append(download(f['download_url'], archive, f['size'], f['computed_md5'], 'md5'))
        dest = ROOT / 'matgpt' / 'checkpoints' / selected[f['name']]
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as z:
            for entry in z.infolist():
                target = (dest / entry.filename).resolve()
                if not target.is_relative_to(dest.resolve()):
                    raise ValueError('Ruta de archivo ZIP no válida')
            if bad := z.testzip():
                raise ValueError(f'CRC incorrecto: {bad}')
            z.extractall(dest)
        print(f'Extraído {f["name"]}', flush=True)
    return dict(repository='devalab/matgpt', revision=rev,
                figshare_article=34330655, selected=selected, files=records)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=['mattext', 'cloud', 'matgpt'],
                        default=['mattext', 'cloud'],
                        help='Selección activa: MatText y CLOUD. MatGPT solo por petición explícita.')
    args = parser.parse_args()
    downloaders = {'mattext': mattext, 'cloud': cloud, 'matgpt': matgpt}
    results = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        tasks = {name: pool.submit(downloaders[name]) for name in dict.fromkeys(args.models)}
        for name, task in tasks.items():
            results[name] = task.result()
            (ROOT / f'{name}-manifest.json').write_text(json.dumps(results[name], indent=2) + '\n')

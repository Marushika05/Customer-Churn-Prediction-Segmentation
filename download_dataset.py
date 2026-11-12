"""Download UCI dataset 502 without overwriting any existing original data."""
from pathlib import Path
import shutil
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent
URL = 'https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip'


def main():
    folder = ROOT/'data/raw'
    folder.mkdir(parents=True, exist_ok=True)
    target = folder/'online_retail_II.xlsx'
    if target.exists():
        print('Workbook already exists; left unchanged:', target)
        return
    archive = folder/'online_retail_ii_uci.zip'
    if not archive.exists():
        partial = archive.with_suffix('.download')
        with urllib.request.urlopen(URL, timeout=120) as response, partial.open('wb') as dest:
            shutil.copyfileobj(response, dest)
        # Validate the archive before making it the canonical download.
        with zipfile.ZipFile(partial) as z:
            if z.testzip() is not None:
                raise ValueError('UCI archive failed CRC validation')
        partial.rename(archive)
    with zipfile.ZipFile(archive) as z:
        member = next(n for n in z.namelist() if n.lower().endswith('.xlsx'))
        with z.open(member) as src, target.open('xb') as dest:
            shutil.copyfileobj(src, dest)
    print('Downloaded complete workbook:', target)


if __name__ == '__main__':
    main()

import re
from pathlib import Path
import zipfile

def parse_idx_table(idx_table_str):
    file_ids = re.findall(r'<Path>(.+)</Path>', idx_table_str)
    file_paths = re.findall(r'<OriginalFileName>(.+)</OriginalFileName>', idx_table_str)
    file_names = [Path(file_path).name for file_path in file_paths]
    file_dict = dict(zip(file_ids, file_names))
    return file_dict

def extract_cag(filepath, target=None):
    if target is None:
        target = Path(filepath).parent
    else:
        target = Path(target)
        if not target.exists():
            target.mkdir()
    with zipfile.ZipFile(filepath, 'r') as zip_ref:
        idx_table = None
        file_list = zip_ref.infolist()
        vk_files = []
        for file_name in file_list:
            with zip_ref.open(file_name) as file:
                magic = file.read(5)
                if magic[:3] not in [b'VK3', b'VK4', b'VK6', b'VK7']:
                    continue
                file_id = zipfile.Path(root=zip_ref, at=file_name.orig_filename).parent.name

                if idx_table is None:
                    for p in zipfile.Path(root=zip_ref, at=file_name.orig_filename).parent.parent.iterdir():
                        if p.is_file():
                            with p.open() as xmlfile:
                                idx_table = parse_idx_table(xmlfile.read())
                file.seek(0)
                with open(target / idx_table[file_id], 'wb') as target_file:
                    target_file.write(file.read())
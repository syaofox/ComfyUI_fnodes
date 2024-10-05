from pathlib import Path

import requests
from tqdm import tqdm


def download_model(model_url, save_loc, model_name):
    if isinstance(save_loc, str):
        save_loc = Path(save_loc)
    save_loc.mkdir(parents=True, exist_ok=True)

    if not (save_loc / model_name).is_file():
        print(f'fnodes: 正在下载模型{model_name}')
        response = requests.get(model_url, stream=True)
        try:
            if response.status_code == 200:
                total_size = int(response.headers.get('content-length', 0))
                block_size = 1024  # 1 Kibibyte

                with (
                    (save_loc / model_name).open('wb') as file,
                    tqdm(
                        desc='下载中',
                        total=total_size,
                        unit='iB',
                        unit_scale=True,
                        unit_divisor=1024,
                    ) as bar,
                ):
                    for data in response.iter_content(block_size):
                        bar.update(len(data))
                        file.write(data)
                print(f'fnodes: 模型{model_name}下载完成')
            else:
                print(f'fnodes: 模型{model_name}下载失败: {response.status_code}')

        except requests.exceptions.RequestException as err:
            print(f'fnodes: 模型{model_name}下载失败: {err}')
            print(f'fnodes: 请从以下链接手动下载: {model_url}')
            print(f'fnodes: 并将其放置在 {save_loc}')
            return False
        except Exception as e:
            print(f'fnodes: 发生意外错误: {e}')
            return False

    return True

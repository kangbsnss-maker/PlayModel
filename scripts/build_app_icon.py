"""Convert the supplied artwork into Windows icon sizes without changing its design."""
from pathlib import Path
from PIL import Image, ImageOps


def main():
    directory = Path(__file__).resolve().parents[1] / 'assets/app-icon'
    with Image.open(directory / '아이콘.png') as original:
        icon = ImageOps.pad(original.convert('RGBA'), (256, 256),
                            method=Image.Resampling.LANCZOS, color=(0, 0, 0, 0))
        icon.save(directory / 'playmodel.ico', sizes=[(n, n) for n in (16, 24, 32, 48, 64, 128, 256)])
        icon.save(directory / 'playmodel.png')
    print('APP_ICON_BUILT')


if __name__ == '__main__':
    main()

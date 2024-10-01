import subprocess

with open('requirements.txt') as f:
    for line in f:
        package = line.strip()
        # Skip empty lines and comments
        if not package or package.startswith('#'):
            continue
        print(f"Attempting to install: {package}")
        try:
            subprocess.check_call(['pip', 'install', package])
            print(f"Successfully installed: {package}")
        except subprocess.CalledProcessError:
            print(f"Failed to install: {package}, skipping...")

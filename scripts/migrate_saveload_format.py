# Migrate saved agent to the new format.

import os
import shutil

def migrate(path, dest_path=None):
    """Migrate a saved agent to the new format.

    Args:
        path: Path to the saved agent.
        dest_path: Path to save the migrated agent. If None, the agent will be saved in the same directory as the old one with _adapt added to the name.
    """
    if dest_path is None:
        dest_path = os.path.join(os.path.dirname(path), os.path.basename(path) + '_adapt')
    print(f"Migrating {path} to {dest_path}...")
    new_name = os.path.basename(dest_path)

    assert os.path.exists(os.path.join(path, "__init__.py")), "Old agent doesn't seem to be in old format; it doesn't have __init__.py file!"

    # 1. copy everything over
    shutil.copytree(path, dest_path)

    # 2. Tweak the code in init.py
    with open(os.path.join(dest_path, "__init__.py"), "r") as f:
        old_init = f.read()
    new_init = old_init.replace("os.path.dirname(__file__)", "os.path.dirname(os.path.dirname(__file__))")
    with open(os.path.join(dest_path, "__init__.py"), "w") as f:
        f.write(new_init)

    # 3 move code and __init__ into subdirectory
    module_name = f'{new_name}_module'
    module_path = os.path.join(dest_path, module_name)
    os.mkdir(module_path)
    shutil.move(os.path.join(dest_path, "code"), os.path.join(module_path, "code"))
    shutil.move(os.path.join(dest_path, "__init__.py"), os.path.join(module_path, "__init__.py"))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Migrate a saved agent to intermediate new format.")
    parser.add_argument("path", help="Path to the saved agent.")
    parser.add_argument("--dest", help="Path to save the migrated agent. If None, the agent will be saved in the same directory as the old one with _adapt added to the name.")
    args = parser.parse_args()
    migrate(args.path, args.dest)
    print("Done!")

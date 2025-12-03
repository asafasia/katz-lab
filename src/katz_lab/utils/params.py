import json
from katz_lab.utils.config import ARGS_PATH


class Params:
    def __init__(self, path=ARGS_PATH):
        self.path = path
        with open(path) as f:
            self.data = json.load(f)

    def _save(self):
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=4)

    def get(self, path, default=None):
        keys = path.split("/")
        cur = self.data
        for k in keys:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                if default is not None:
                    return default
                raise KeyError(f"Key not found in JSON: '{k}' in path '{path}'")
        return cur

    def set(self, path, value):
        keys = path.split("/")
        cur = self.data

        # Navigate to the parent dict
        for k in keys[:-1]:
            if k not in cur or not isinstance(cur[k], dict):
                raise KeyError(
                    f"Cannot update: missing parent key '{k}' while accessing '{path}'"
                )
            cur = cur[k]

        final_key = keys[-1]
        if final_key not in cur:
            raise KeyError(
                f"Cannot update: final key '{final_key}' does not exist in '{path}'"
            )

        # Perform update
        cur[final_key] = value
        self._save()

        # ✔ nice confirmation print
        print(f"Updated '{path}' → {value}")

    def __getitem__(self, path):
        return self.get(path)

    def __setitem__(self, path, value):
        self.set(path, value)


args = Params()


if __name__ == "__main__":

    args["q10/qubit/reset_gate/length"] = 70  # OK

    # p["q10/qubit/does_not_exist"] = 10  # ❌ raises KeyError

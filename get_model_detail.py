#!/usr/bin/env python3

import sys
import os

# Optional imports (lazy mindset: only fail when needed)
try:
    from stai_mpu import stai_mpu_network
except ImportError:
    stai_mpu_network = None

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    tflite = None


# ============================================
# BASE CLASS (because structure matters)
# ============================================
class BaseModelInspector:
    def inspect(self):
        raise NotImplementedError("inspect() must be implemented")


# ============================================
# NB MODEL INSPECTOR
# ============================================
class NBModelInspector(BaseModelInspector):
    def __init__(self, model_path):
        if stai_mpu_network is None:
            raise ImportError("stai_mpu not available")

        self.model = stai_mpu_network(
            model_path=model_path,
            use_hw_acceleration=True
        )

    def dump_tensor_info(self, info, name):
        print(f"\n========== {name} RAW OBJECT ==========")
        print(info)

        print(f"\n---- {name} ALL ATTRIBUTES ----")
        for attr in dir(info):
            if attr.startswith("__"):
                continue

            try:
                value = getattr(info, attr)

                if callable(value):
                    try:
                        result = value()
                        print(f"{attr}() : {result}")
                    except Exception:
                        print(f"{attr}() : <call failed>")
                else:
                    print(f"{attr} : {value}")

            except Exception:
                print(f"{attr} : <error>")

    def inspect(self):
        print("\n========= NB MODEL LOADING =========")

        input_infos = self.model.get_input_infos()
        print("\n========= INPUT DETAILS =========")
        print("Number of inputs:", len(input_infos))

        for i, info in enumerate(input_infos):
            print(f"\n[Input {i}]")
            print("Shape :", info.get_shape())
            print("Dtype :", info.get_dtype())
            self.dump_tensor_info(info, f"INPUT {i}")

        output_infos = self.model.get_output_infos()
        print("\n========= OUTPUT DETAILS =========")
        print("Number of outputs:", len(output_infos))

        for i, info in enumerate(output_infos):
            print(f"\n[Output {i}]")
            print("Shape :", info.get_shape())
            print("Dtype :", info.get_dtype())
            self.dump_tensor_info(info, f"OUTPUT {i}")


# ============================================
# TFLITE MODEL INSPECTOR
# ============================================
class TFLiteModelInspector(BaseModelInspector):
    def __init__(self, model_path):
        if tflite is None:
            raise ImportError("tflite_runtime not available")

        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

    def inspect(self):
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        print("\n================ INPUT DETAILS ================")
        for i, inp in enumerate(input_details):
            print(f"[Input {i}]")
            print(" index        :", inp["index"])
            print(" shape        :", inp["shape"])
            print(" dtype        :", inp["dtype"])
            print(" quantization :", inp.get("quantization", "N/A"))
            print(" name         :", inp.get("name", "N/A"))
            print("---------------------------------------------")

        print("\n================ OUTPUT DETAILS ================")
        for i, out in enumerate(output_details):
            print(f"[Output {i}]")
            print(" index        :", out["index"])
            print(" shape        :", out["shape"])
            print(" dtype        :", out["dtype"])
            print(" quantization :", out.get("quantization", "N/A"))
            print(" name         :", out.get("name", "N/A"))
            print("---------------------------------------------")


# ============================================
# FACTORY (the brain, finally)
# ============================================
class ModelInspectorFactory:
    @staticmethod
    def create(model_path):
        ext = os.path.splitext(model_path)[1].lower()

        if ext == ".nb":
            return NBModelInspector(model_path)

        elif ext == ".tflite":
            return TFLiteModelInspector(model_path)

        else:
            raise ValueError(f"Unsupported model format: {ext}")


# ============================================
# ENTRY POINT
# ============================================
def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <model.(nb|tflite)>")
        sys.exit(1)

    model_path = sys.argv[1]

    try:
        inspector = ModelInspectorFactory.create(model_path)
        inspector.inspect()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

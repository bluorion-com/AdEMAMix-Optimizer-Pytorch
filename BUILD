load("@aspect_rules_py//py:defs.bzl", "py_library", "py_test")

py_library(
    name = "AdEMAMix",
    srcs = ["AdEMAMix.py"],
    deps = [
        "@pip//numpy:pkg",
        "@pip//torch:pkg",
    ],
)

py_test(
    name = "AdEMAMix_test",
    srcs = ["AdEMAMix_test.py"],
    deps = [
        ":AdEMAMix",
        "@pip//absl_py:pkg",
    ],
)

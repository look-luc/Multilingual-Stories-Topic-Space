from datasets import load_dataset

from src.data.lang_codes import DEEPON_LANG_CODE, DXNIZ_LANG_CODE, WAXAL_LANG_CODE


def get_some_data():
    ds_deep = load_dataset(
        "deeponh/multilingual-tinystories", streaming=True
    )["train"]
    ds_deep_as = ds_deep.filter(
        lambda example: DEEPON_LANG_CODE.get("Assamese") in example["index"]
    )
    ds_dxniz = load_dataset(
        "Dxniz/TinyStories-Multilingual", streaming=True
    )["train"]
    ds_dxniz_viet = ds_dxniz.filter(
        lambda example: DXNIZ_LANG_CODE.get("Vietnamese") in example["language_code"]
    )

    ds_waxal_Acholi = load_dataset(
        "google/WaxalNLP", f"{WAXAL_LANG_CODE.get("Acholi")}_asr", streaming=True
    )["train"]

    return (
        ds_deep_as,
        ds_dxniz_viet,
        ds_waxal_Acholi
    )

def get_all_data():
    pass

from datasets import load_dataset

from src.data.lang_codes import DEEPON_LANG_CODE, DXNIZ_LANG_CODE, WAXAL_LANG_CODE


def get_some_data(limit=30):
    assamese_code = DEEPON_LANG_CODE.get("Assamese")
    ds_deep = load_dataset(
        "deeponh/multilingual-tinystories",
        split=assamese_code,
        streaming=True
    )
    ds_deep_as = [ex["text"] for ex in ds_deep.take(limit)]

    ds_dxniz = load_dataset("Dxniz/TinyStories-Multilingual", streaming=True)["train"]
    ds_dxniz_viet = [
        ex["output"] for ex in ds_dxniz.filter(
            lambda example: DXNIZ_LANG_CODE.get("Vietnamese") in example["language_code"]
        ).take(limit)
    ]

    ds_waxal_acholi = load_dataset(
        "google/WaxalNLP", f"{WAXAL_LANG_CODE.get('Acholi')}_asr", streaming=True
    )["train"]
    ds_waxal_acholi = [
        ex["transcription"] for ex in ds_waxal_acholi.take(limit)
    ]

    return {
        "Assamese": ds_deep_as,
        "Vietnamese": ds_dxniz_viet,
        "Acholi": ds_waxal_acholi
    }

def get_all_data():
    pass

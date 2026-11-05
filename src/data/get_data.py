from datasets import Audio, concatenate_datasets, load_dataset

from src.data.lang_codes import DEEPON_LANG_CODE, DXNIZ_LANG_CODE, WAXAL_LANG_CODE


def get_some_data(limit: int = 100):
    ds_indic_tinystories = load_dataset(
        "deeponh/multilingual-tinystories",
        split=DEEPON_LANG_CODE.get("Assamese"),
        streaming=True
    ).select_columns(["text"])

    ds_tinystories_multi = load_dataset(
        "Dxniz/TinyStories-Multilingual",
        split="train",
        streaming=True
    ).filter(lambda x: x["language_code"] == DXNIZ_LANG_CODE.get("Vietnamese")) \
     .select_columns(["output"]) \
     .rename_column("output", "text")

    ds_waxal = load_dataset(
        "google/WaxalNLP",
        f"{WAXAL_LANG_CODE.get('Acholi')}_asr",
        split="train",
        streaming=True
    )
    if "audio" in ds_waxal.features:
        ds_waxal = ds_waxal.cast_column("audio", Audio(decode=False))

    ds_waxal = ds_waxal.select_columns(["transcription"]).rename_column("transcription", "text")

    return {
        "Assamese": [ex["text"] for ex in ds_indic_tinystories.take(limit)],
        "Vietnamese": [ex["text"] for ex in ds_tinystories_multi.take(limit)],
        "Acholi": [ex["text"] for ex in ds_waxal.take(limit)],
    }

def get_all_data():
    waxal_datasets = []
    print("getting languages from Google/Waxal")
    for waxal_lang, waxal_code in WAXAL_LANG_CODE.values():
        print(f"getting {waxal_lang} data...")
        ds_waxal = load_dataset(
            "google/WaxalNLP",
            f"{WAXAL_LANG_CODE.get(waxal_code)}_asr",
            split="train",
            streaming=True
        )
        if "audio" in ds_waxal.features:
            ds_waxal = ds_waxal.cast_column("audio", Audio(decode=False))

        ds_waxal = ds_waxal.select_columns(["transcription"]).rename_column("transcription", "text")
        waxal_datasets.append(ds_waxal)

    dxniz_datasets = []
    for dxniz_lang in DXNIZ_LANG_CODE.keys():
        ds_tinystories_multi = load_dataset(
            "Dxniz/TinyStories-Multilingual",
            split="train",
            streaming=True
        ).filter(lambda x: x["language_code"] == DXNIZ_LANG_CODE.get(dxniz_lang))\
        .select_columns(["output"])\
        .rename_column("output", "text")

        dxniz_datasets.append(ds_tinystories_multi)

    deepon_datasets = []
    for deepon_lang in DEEPON_LANG_CODE.keys():
        ds_indic_tinystories = load_dataset(
            "deeponh/multilingual-tinystories",
            split=DEEPON_LANG_CODE.get(deepon_lang),
            streaming=True
        ).select_columns(["text"])
        deepon_datasets.append(ds_indic_tinystories)

    all_langs = concatenate_datasets([waxal_datasets, dxniz_datasets, deepon_datasets])

    return all_langs

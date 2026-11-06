from datasets import load_dataset

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
    ).select_columns(["transcription"]).rename_column("transcription", "text")

    return {
        "Assamese": [ex["text"] for ex in ds_indic_tinystories.take(limit)],
        "Vietnamese": [ex["text"] for ex in ds_tinystories_multi.take(limit)],
        "Acholi": [ex["text"] for ex in ds_waxal.take(limit)],
    }

def get_all_data():
    data_dict = {}

    print("Getting languages from Google/Waxal...")
    for waxal_lang, waxal_code in WAXAL_LANG_CODE.items():
        print(f"Getting {waxal_lang} data...")
        ds_waxal = load_dataset(
            "google/WaxalNLP",
            f"{waxal_code}_asr",
            split="train",
            streaming=True,
            columns=["transcription"]
        ).rename_column("transcription", "text")

        data_dict[waxal_lang] = [ex["text"] for ex in ds_waxal]

    print("Getting languages from Dxniz/TinyStories-Multilingual...")
    for dxniz_lang, dxniz_code in DXNIZ_LANG_CODE.items():
        print(f"Getting {dxniz_lang} data...")
        ds_tinystories_multi = load_dataset(
            "Dxniz/TinyStories-Multilingual",
            split="train",
            streaming=True
        ).filter(lambda x: x["language_code"] == dxniz_code)\
         .select_columns(["output"])\
         .rename_column("output", "text")

        data_dict[dxniz_lang] = [ex["text"] for ex in ds_tinystories_multi]

    print("Getting languages from deeponh/multilingual-tinystories...")
    for deepon_lang, deepon_split in DEEPON_LANG_CODE.items():
        print(f"Getting {deepon_lang} data...")
        ds_indic_tinystories = load_dataset(
            "deeponh/multilingual-tinystories",
            split=deepon_split,
            streaming=True
        ).select_columns(["text"])

        data_dict[deepon_lang] = [ex["text"] for ex in ds_indic_tinystories]

    return data_dict

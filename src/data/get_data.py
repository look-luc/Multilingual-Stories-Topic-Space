from datasets import load_dataset

from src.data.lang_codes import WAXAL_LANG_CODE


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

    amazight = load_dataset(
        "Tamazight-NLP/1002-Amazigh-Proverbs",
        split="train",
        streaming=True,
        columns="amazight"
    ).rename_column("amazight", "text")

    data_dict["Amazight"] = [ex["text"] for ex in amazight]

    ds_yor = load_dataset(
        "saaga/yoruba-cultural-reasoning-blindspots",
        split="train",
        streaming=True,
        columns="input"
    ).rename_column("input", "text")

    data_dict["Yoruba cultural"] = [ex["text"] for ex in ds_yor]

    ds_yor_prov = load_dataset(
        "mxronga/yoruba-proverbs-parallel-corpora",
        split="train",
        streaming=True,
        columns="yoruba"
    ).rename_column("yoruba", "text")

    data_dict["Yoruba proverb"] = [ex["text"] for ex in ds_yor_prov]

    ds_ilubench = load_dataset(
        "UUAMNI/ilubench",
        "probes",
        streaming=True,
        columns="proverb_ig"
    ).rename_column("proverb_ig", "text")

    data_dict["Igbo"] = [ex["text"] for ex in ds_ilubench]

    return data_dict

from datasets import concatenate_datasets, load_dataset

from src.data.lang_codes import RIDDLE_LANG_CODE, WAXAL_LANG_CODE


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
        columns=["amazight"]
    ).rename_column("amazight", "text")

    data_dict["Amazight"] = [ex["text"] for ex in amazight]

    ds_yor = load_dataset(
        "saaga/yoruba-cultural-reasoning-blindspots",
        split="train",
        streaming=True,
        columns=["input"]
    ).rename_column("input", "text")

    ds_yor_prov = load_dataset(
        "mxronga/yoruba-proverbs-parallel-corpora",
        split="train",
        streaming=True,
        columns=["yoruba"]
    ).rename_column("yoruba", "text")

    yoruba = concatenate_datasets([ds_yor, ds_yor_prov])

    data_dict["Yoruba proverb"] = [ex["text"] for ex in yoruba]

    ds_ilubench = load_dataset(
        "UUAMNI/ilubench",
        "probes",
        streaming=True,
        columns=["proverb_ig"]
    ).rename_column("proverb_ig", "text")

    data_dict["Igbo"] = [ex["text"] for ex in ds_ilubench]

    ds_luyia = load_dataset(
        "mamakobe/luhya-multilingual-dataset",
        split="train",
        streaming=True,
        columns=["luhya_text"]
    ).rename_column("luhya_text", "text")

    data_dict["Luyia"] = [ex["text"] for ex in ds_luyia]

    ds_rukiga = load_dataset(
        "rukiga-yoruba-datacuration/rukiga-yoruba-short-text-proverbs",
        "cgg",
        split="train",
        streaming=True
    )
    ds_rukiga = ds_rukiga.filter(lambda example: example["domain"] == "proverb")["text"]

    data_dict["Rukiga"] = [ex["text"] for ex in ds_rukiga]

    ds_riddles = load_dataset("CohereLabsCommunity/CulturalRiddles", "riddles")
    for ex in ds_riddles:
        lang_code = str(ex.get("language")).strip().lower()
        if lang_code not in RIDDLE_LANG_CODE:
            continue

        riddle_lang = RIDDLE_LANG_CODE[lang_code]
        riddle_text = ex.get("riddle_original")
        if riddle_text:
            data_dict.setdefault(riddle_lang, []).append(riddle_text)

    ds_ticuna = load_dataset(
        "aimeri/ticuna-spanish-portuguese",
        "monolingual",
        streaming=True,
        columns=["source"]
    ).rename_column("source", "text")

    data_dict["Ticuna"] = [ex["text"] for ex in ds_ticuna]

    return data_dict

from datasets import Audio, concatenate_datasets, load_dataset

from src.data.lang_codes import RIDDLE_LANG_CODE, WAXAL_LANG_CODE


def get_all_data():
    data_dict = {}

    print("Getting languages from Google/Waxal...")
    for waxal_lang, waxal_code in WAXAL_LANG_CODE.items():
        print(f"Getting {waxal_lang} data...")
        try:
            ds_waxal = (
                load_dataset(
                    "google/WaxalNLP",
                    f"{waxal_code}_asr",
                    split="train",
                    streaming=True,
                )
                .cast_column("audio", Audio(decode=False))
                .select_columns(["transcription"])
            )
            data_dict[waxal_lang] = [
                ex["transcription"]
                for ex in ds_waxal
                if isinstance(ex, dict) and ex.get("transcription")
            ]
        except Exception as exc:
            print(f"Warning: skipping Waxal {waxal_lang} ({waxal_code}_asr) due to error: {exc}")
            data_dict[waxal_lang] = []

    amazight = load_dataset(
        "Tamazight-NLP/1002-Amazigh-Proverbs",
        split="train",
        streaming=True,
    )
    data_dict["Amazight"] = [
        ex["amazight"] for ex in amazight if isinstance(ex, dict) and ex.get("amazight")
    ]

    ds_yor = load_dataset(
        "saaga/yoruba-cultural-reasoning-blindspots",
        split="train",
        streaming=True,
    ).rename_column("input", "text")

    ds_yor_prov = load_dataset(
        "mxronga/yoruba-proverbs-parallel-corpora",
        split="train",
        streaming=True,
    ).rename_column("yoruba", "text")

    yoruba = concatenate_datasets([ds_yor, ds_yor_prov])
    data_dict["Yoruba proverb"] = [
        ex["text"] for ex in yoruba if isinstance(ex, dict) and ex.get("text")
    ]

    ds_ilubench = load_dataset(
        "UUAMNI/ilubench",
        "probes",
        streaming=True,
    )
    igbo_proverbs = []
    for _, ds_split in ds_ilubench.items():
        igbo_proverbs.extend(
            ex["proverb_ig"]
            for ex in ds_split
            if isinstance(ex, dict) and ex.get("proverb_ig")
        )
    data_dict["Igbo"] = igbo_proverbs

    ds_luyia = load_dataset(
        "mamakobe/luhya-multilingual-dataset",
        split="train",
        streaming=True,
    )
    data_dict["Luyia"] = [
        ex["luhya_text"] for ex in ds_luyia if isinstance(ex, dict) and ex.get("luhya_text")
    ]

    ds_rukiga = load_dataset(
        "rukiga-yoruba-datacuration/rukiga-yoruba-short-text-proverbs",
        "cgg",
        split="train",
        streaming=True,
    )
    ds_rukiga = ds_rukiga.filter(lambda example: example["domain"] == "proverb")
    data_dict["Rukiga"] = [
        ex["text"] for ex in ds_rukiga if isinstance(ex, dict) and ex.get("text")
    ]

    ds_riddles = load_dataset("CohereLabsCommunity/CulturalRiddles", "riddles")
    for ex in ds_riddles:
        if not isinstance(ex, dict):
            continue
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
    )
    data_dict["Ticuna"] = [
        ex["source"] for ex in ds_ticuna if isinstance(ex, dict) and ex.get("source")
    ]

    return data_dict

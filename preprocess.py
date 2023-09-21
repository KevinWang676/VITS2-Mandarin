import os
import argparse
import json
import sys
sys.setrecursionlimit(500000)  # Fix the error message of RecursionError: maximum recursion depth exceeded while calling a Python object.  You can change the number as you want.

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--add_auxiliary_data", type=bool, default="False", help="Whether to add extra data as fine-tuning helper")
    parser.add_argument("--languages", default="C")
    args = parser.parse_args()
    if args.languages == "CJE":
        langs = ["[ZH]", "[JA]", "[EN]"]
    elif args.languages == "CJ":
        langs = ["[ZH]", "[JA]"]
    elif args.languages == "C":
        langs = ["[ZH]"]
    new_annos = []
    # Source 1: transcribed short audios
    if os.path.exists("./filelists/short_character_anno.list"):
        with open("./filelists/short_character_anno.list", 'r', encoding='utf-8') as f:
            short_character_anno = f.readlines()
            new_annos += short_character_anno

    # Get all speaker names
    speakers = []
    for line in new_annos:
        path, speaker, text = line.split("|")
        if speaker not in speakers:
            speakers.append(speaker)
    assert (len(speakers) != 0), "No audio file found. Please check your uploaded file structure."
    if True:
        # Do not add extra helper data
        # STEP 1: modify config file
        with open("./configs/finetune_speaker.json", 'r', encoding='utf-8') as f:
            hps = json.load(f)

        # assign ids to new speakers
        speaker2id = {}
        for i, speaker in enumerate(speakers):
            speaker2id[speaker] = i
        # modify n_speakers
        hps['data']["n_speakers"] = len(speakers)
        # overwrite speaker names
        hps['speakers'] = speaker2id
        hps['train']['log_interval'] = 10
        hps['train']['eval_interval'] = 100
        hps['train']['batch_size'] = 16
        hps['data']['training_files'] = "final_annotation_train.txt"
        hps['data']['validation_files'] = "final_annotation_val.txt"
        # save modified config
        with open("./configs/modified_finetune_speaker.json", 'w', encoding='utf-8') as f:
            json.dump(hps, f, indent=2)

        # STEP 2: clean annotations, replace speaker names with assigned speaker IDs
        import text

        cleaned_new_annos = []
        for i, line in enumerate(new_annos):
            path, speaker, txt = line.split("|")
            if len(txt) > 150:
                continue
            cleaned_text = text._clean_text(txt, hps['data']['text_cleaners']).replace("[ZH]", "")
            cleaned_text += "\n" if not cleaned_text.endswith("\n") else ""
            cleaned_new_annos.append(path + "|0|" + cleaned_text)

        final_annos = cleaned_new_annos
        # save annotation file
        with open("./filelists/final_annotation_train.txt", 'w', encoding='utf-8') as f:
            for line in final_annos:
                f.write(line)
        # save annotation file for validation
        with open("./filelists/final_annotation_val.txt", 'w', encoding='utf-8') as f:
            for line in cleaned_new_annos:
                f.write(line)
        print("finished")

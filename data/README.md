# Data access: no dataset payloads are included

The project preserves immutable document-level files named `panyo_original_adp_docsplit_{train,dev,test}.conllu`. Use `results/public_camera_ready/split_checksums.json` to verify authorized copies. TRAIN has 6,579 documents / 45,450 sentences / 788,802 tokens; DEV 819 / 5,690 / 98,566; TEST 835 / 5,701 / 98,529. Tokens mean real integer-ID CoNLL-U words, excluding ranges/empty nodes.

The mapping `data/processed/panyo_jsonl_doc_sentence_map.csv` is private because it links source records. Paragraph-derived IDs do not imply independent articles. Never regenerate or substitute older sentence-level splits for published evaluation.

Upstream source: https://github.com/MrBananaHuman/PangyoCorpora . Its README states MIT licensing; that is not independent privacy clearance. Original hosted ChatGPT-assisted silver sessions are not recoverable. Downloading upstream text does not provide this project's exact canonical labels and partitions.

UD inputs must be separately obtained from the official English GUM and Korean KAIST repositories under their own licenses. Original code expects `data/en_gum-ud-train.conllu` and `data/ko_kaist-ud-train.conllu`; the public baseline adapter also accepts arbitrary explicit paths. Exact original treebank versions require matching original hashes, not filename equivalence. No download wrapper invents the missing version pin.

No verified public distribution of frozen splits, references, models or predictions is available in this release. Access/redistribution requires separate maintainer clearance; no public access promise is made. Human source data, independent pre-adjudication annotations, answer keys and the operative guideline bundle are withheld. Generic historical annotation scaffolds are deliberately omitted and must not substitute for the frozen human guideline.

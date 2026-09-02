---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:17560
- loss:MultipleNegativesRankingLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: 'Skills: Access management software, Adobe ActionScript, Adobe
    After Effects, Advanced business application programming ABAP, AirMagnet Enterprise,
    AJAX, Amazon DynamoDB, Amazon Elastic Compute Cloud EC2, Amazon Redshift, Amazon
    Simple Storage Service S3, Amazon Web Services AWS CloudFormation, Amazon Web
    Services AWS software, Ansible software, Apache Ant, Apache Cassandra, Apache
    Groovy, Apache Hadoop, Apache Hive, Apache HTTP Server, Apache Kafka. Education:
    8.27. Experience: 8.3.'
  sentences:
  - 'Job Description: Apply makeup to performers to reflect period, setting, and situation
    of their role.'
  - 'Job Description: Help electricians by performing duties requiring less skill.
    Duties include using, supplying, or holding materials or tools, and cleaning work
    area and equipment.'
  - 'Job Description: Design and implement computer and information networks, such
    as local area networks (LAN), wide area networks (WAN), intranets, extranets,
    and other data communications networks. Perform network modeling, analysis, and
    planning, including analysis of capacity needs for network infrastructures. May
    also design network and computer security measures. May research and recommend
    network and data communications hardware and software.'
- source_sentence: 'Skills: Microsoft Access, Microsoft Excel, Microsoft Word, Electronic
    medical record EMR software, Microsoft Office software, MEDITECH software, Web
    browser software. Education: 8.38. Experience: 10.7.'
  sentences:
  - 'Job Description: Assist other social and human service providers in providing
    client services in a wide variety of fields, such as psychology, rehabilitation,
    or social work, including support for families. May assist clients in identifying
    and obtaining available benefits and social and community services. May assist
    social workers with developing, organizing, and conducting programs to prevent
    and resolve problems relevant to substance abuse, human relationships, rehabilitation,
    or dependent care.'
  - 'Job Description: Perform duties related to the purchase, sale, or holding of
    securities. Duties include writing orders for stock purchases or sales, computing
    transfer taxes, verifying stock transactions, accepting and delivering securities,
    tracking stock price fluctuations, computing equity, distributing dividends, and
    keeping records of daily transactions and holdings.'
  - 'Job Description: Plan, organize, and conduct long-distance travel, tours, and
    expeditions for individuals and groups.'
- source_sentence: 'Skills: Microsoft Outlook, Microsoft Excel, Machine operation
    software, Microsoft Office software. Education: 8.59. Experience: 9.3.'
  sentences:
  - 'Job Description: Take orders and serve food and beverages to patrons at tables
    in dining establishment.'
  - 'Job Description: Operate or control an entire process or system of machines,
    often through the use of control boards, to transfer or treat water or wastewater.'
  - 'Job Description: Feed materials into or remove materials from machines or equipment
    that is automatic or tended by other workers.'
- source_sentence: 'Skills: Microsoft Excel, Automated inventory software, Computerized
    numerical control CNC software, SAP software. Education: 8.38. Experience: 7.7.'
  sentences:
  - 'Job Description: Operate industrial trucks or tractors equipped to move materials
    around a warehouse, storage yard, factory, construction site, or similar location.'
  - 'Job Description: Set up, operate, or tend drilling machines to drill, bore, ream,
    mill, or countersink metal or plastic work pieces.'
  - 'Job Description: Diagnose, manage, and treat conditions and diseases of the human
    eye and visual system. Examine eyes and visual system, diagnose problems or impairments,
    prescribe corrective lenses, and provide treatment. May prescribe therapeutic
    drugs to treat specific eye conditions.'
- source_sentence: 'Skills: Automatic Data Processing AdvancedMD EHR, Email software,
    Allscripts PM, GraphPad Software GraphPad Prism, athenahealth athenaCollector,
    GalacTek ECLIPSE, Crowell Systems Medformix, Electronic Medical Records and Electronic
    Health Records Software IMS for Allergists, CareCloud Central, IOS Health Systems
    Medios EHR, Benchmark Systems Benchmark Clinical EHR, Greenway Medical Technologies
    PrimeSUITE, GE Healthcare Centricity Practice Solution, FlowJo, HealthFusion MediTouch,
    Bizmatics PrognoCIS EMR, eClinicalWorks EHR software, Cerner PowerWorks Practice
    Management, Epic Practice Management, Kareo Practice Management. Education: 8.24.
    Experience: 11.0.'
  sentences:
  - 'Job Description: Research, design, plan, or perform engineering duties in the
    prevention, control, and remediation of environmental hazards using various engineering
    disciplines. Work may include waste treatment, site remediation, or pollution
    control technology.'
  - 'Job Description: Pilot and navigate the flight of fixed-wing aircraft, usually
    on scheduled air carrier routes, for the transport of passengers and cargo. Requires
    Federal Air Transport certificate and rating for specific aircraft type used.
    Includes regional, national, and international airline pilots and flight instructors
    of airline pilots.'
  - 'Job Description: Diagnose, treat, and help prevent allergic diseases and disease
    processes affecting the immune system.'
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) <!-- at revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
- **Supported Modality:** Text
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'transformer_task': 'feature-extraction', 'modality_config': {'text': {'method': 'forward', 'method_output_name': 'last_hidden_state'}}, 'module_output_name': 'token_embeddings', 'architecture': 'BertModel'})
  (1): Pooling({'embedding_dimension': 384, 'pooling_mode': 'mean', 'include_prompt': True})
  (2): Normalize({})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```
Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'Skills: Automatic Data Processing AdvancedMD EHR, Email software, Allscripts PM, GraphPad Software GraphPad Prism, athenahealth athenaCollector, GalacTek ECLIPSE, Crowell Systems Medformix, Electronic Medical Records and Electronic Health Records Software IMS for Allergists, CareCloud Central, IOS Health Systems Medios EHR, Benchmark Systems Benchmark Clinical EHR, Greenway Medical Technologies PrimeSUITE, GE Healthcare Centricity Practice Solution, FlowJo, HealthFusion MediTouch, Bizmatics PrognoCIS EMR, eClinicalWorks EHR software, Cerner PowerWorks Practice Management, Epic Practice Management, Kareo Practice Management. Education: 8.24. Experience: 11.0.',
    'Job Description: Diagnose, treat, and help prevent allergic diseases and disease processes affecting the immune system.',
    'Job Description: Pilot and navigate the flight of fixed-wing aircraft, usually on scheduled air carrier routes, for the transport of passengers and cargo. Requires Federal Air Transport certificate and rating for specific aircraft type used. Includes regional, national, and international airline pilots and flight instructors of airline pilots.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.6349, 0.0472],
#         [0.6349, 1.0000, 0.0403],
#         [0.0472, 0.0403, 1.0000]])
```
<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 17,560 training samples
* Columns: <code>sentence_0</code> and <code>sentence_1</code>
* Approximate statistics based on the first 100 samples:
  |          | sentence_0                                                                         | sentence_1                                                                          |
  |:---------|:-----------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|
  | type     | string                                                                             | string                                                                              |
  | modality | text                                                                               | text                                                                                |
  | details  | <ul><li>min: 19 tokens</li><li>mean: 75.9 tokens</li><li>max: 147 tokens</li></ul> | <ul><li>min: 10 tokens</li><li>mean: 42.71 tokens</li><li>max: 109 tokens</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | sentence_1                                                                                                                                                                                                                                                                      |
  |:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>Skills: Database software, Garment tracking software, Microsoft Excel, Microsoft Office software, Microsoft Outlook, Microsoft Word, Web browser software. Education: 8.03. Experience: 13.3.</code>                                                                                                                                                                                                                                                                                                                                                                                            | <code>Job Description: Select, fit, and take care of costumes for cast members, and aid entertainers. May assist with multiple costume changes during performances.</code>                                                                                                      |
  | <code>Skills: DesignWare 3D EyeWitness, Integrated Automated Fingerprint Identification System IAFIS, Crime mapping software, Computer aided dispatch software, Microsoft Access, National Integrated Ballistics Information Network NIBIN, Microsoft Active Server Pages ASP, Microsoft Office software, Corel WordPerfect Office Suite, Microsoft Word, Microsoft Internet Explorer, Scheduling software, Law enforcement information databases, Microsoft PowerPoint, National Crime Information Center (NCIC) database, Email software, Microsoft Visio. Education: 8.47. Experience: 7.9.</code> | <code>Job Description: Directly supervise and coordinate activities of members of police force.</code>                                                                                                                                                                          |
  | <code>Skills: Adobe ActionScript, C++, C, Adobe Creative Cloud software, Adobe After Effects, Autodesk Scaleform, Blackboard software, Graphical user interface GUI design software, Git, Autodesk 3ds Max, Adobe Photoshop, Extensible markup language XML, Advanced business application programming ABAP, Adobe Illustrator, C#, Atlassian JIRA, Autodesk Maya. Education: 8.33. Experience: 6.1.</code>                                                                                                                                                                                           | <code>Job Description: Design core features of video games. Specify innovative game and role-play mechanics, story lines, and character biographies. Create and maintain design documentation. Guide and collaborate with production staff to produce games as designed.</code> |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim",
      "gather_across_devices": false,
      "directions": [
          "query_to_doc"
      ],
      "partition_mode": "joint",
      "hardness_mode": null,
      "hardness_strength": 0.0
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `num_train_epochs`: 2
- `per_device_eval_batch_size`: 16
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `per_device_train_batch_size`: 16
- `num_train_epochs`: 2
- `max_steps`: -1
- `learning_rate`: 5e-05
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_steps`: 0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `optim_target_modules`: None
- `gradient_accumulation_steps`: 1
- `average_tokens_across_devices`: True
- `max_grad_norm`: 1
- `label_smoothing_factor`: 0.0
- `bf16`: False
- `fp16`: False
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `use_cache`: False
- `neftune_noise_alpha`: None
- `torch_empty_cache_steps`: None
- `auto_find_batch_size`: False
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `include_num_input_tokens_seen`: no
- `log_level`: passive
- `log_level_replica`: warning
- `disable_tqdm`: False
- `project`: huggingface
- `trackio_space_id`: None
- `trackio_bucket_id`: None
- `trackio_static_space_id`: None
- `per_device_eval_batch_size`: 16
- `prediction_loss_only`: True
- `eval_on_start`: False
- `eval_do_concat_batches`: True
- `eval_use_gather_object`: False
- `eval_accumulation_steps`: None
- `include_for_metrics`: []
- `batch_eval_metrics`: False
- `save_only_model`: False
- `save_on_each_node`: False
- `enable_jit_checkpoint`: False
- `push_to_hub`: False
- `hub_private_repo`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_always_push`: False
- `hub_revision`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `restore_callback_states_from_checkpoint`: False
- `full_determinism`: False
- `seed`: 42
- `data_seed`: None
- `use_cpu`: False
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `dataloader_prefetch_factor`: None
- `dataloader_multiprocessing_context`: None
- `dataloader_in_order`: True
- `remove_unused_columns`: True
- `label_names`: None
- `train_sampling_strategy`: random
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `ddp_static_graph`: None
- `ddp_backend`: None
- `ddp_timeout`: 1800
- `fsdp`: None
- `fsdp_config`: None
- `deepspeed`: None
- `debug`: []
- `skip_memory_metrics`: True
- `do_predict`: False
- `resume_from_checkpoint`: None
- `local_rank`: -1
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin
- `router_mapping`: {}
- `learning_rate_mapping`: {}
- `warmup_ratio`: None

</details>

### Training Logs
| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 0.4554 | 500  | 0.8878        |
| 0.9107 | 1000 | 0.2931        |
| 1.3661 | 1500 | 0.1668        |
| 1.8215 | 2000 | 0.1299        |


### Training Time
- **Training**: 2.1 hours

### Framework Versions
- Python: 3.12.10
- Sentence Transformers: 5.7.0
- Transformers: 5.15.0
- PyTorch: 2.8.0+cpu
- Accelerate: 1.14.0
- Datasets: 5.0.1
- Tokenizers: 0.22.2

## Additional Resources

- [Training and Finetuning Embedding Models with Sentence Transformers](https://huggingface.co/blog/train-sentence-transformers): the end-to-end guide for training or finetuning Sentence Transformer models.
- [Introduction to Matryoshka Embedding Models](https://huggingface.co/blog/matryoshka): variable-size embeddings that can be truncated with minimal quality loss.
- [Binary and Scalar Embedding Quantization for Significantly Faster & Cheaper Retrieval](https://huggingface.co/blog/embedding-quantization): post-training compression of embedding vectors.
- [Multimodal Embedding & Reranker Models with Sentence Transformers](https://huggingface.co/blog/multimodal-sentence-transformers): use text, image, audio, and video models through the same API.
- [Training and Finetuning Multimodal Embedding & Reranker Models with Sentence Transformers](https://huggingface.co/blog/train-multimodal-sentence-transformers): train multimodal embedding models, with a Visual Document Retrieval walkthrough.

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{oord2019representationlearningcontrastivepredictive,
      title={Representation Learning with Contrastive Predictive Coding},
      author={Aaron van den Oord and Yazhe Li and Oriol Vinyals},
      year={2019},
      eprint={1807.03748},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/1807.03748},
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->
Data construction code for [`Schema-adaptable Knowledge Graph Construction`]()

Note: First of all, make sure your current path is **`*/AdaKGC/dataset_construct`**

# Quick links

- [Quick links](#quick-links)
- [Obtain Raw Dataset](#obtain-raw-dataset)
    - [Few-NERD](#few-nerd)
    - [NYT](#nyt)
    - [ACE05-Evt](#ace05-evt)
- [Convert Raw Dataset to Iteration Dataset](#convert-raw-dataset-to-iteration-dataset)
  - [Details of Dataset Conversion](#details-of-dataset-conversion)
      - [Example of Dataset Configuration file](#example-of-dataset-configuration-file)
  - [File Name Format](#file-name-format)
      - [Schema Hierarchy](#schema-hierarchy)
      - [Segmentation Mode](#segmentation-mode)
      - [Iteration change](#iteration-change)
- [Data Format](#data-format)
      - [Example of instance](#example-of-instance)
- [Acknowledgement](#acknowledgement)

# Obtain Raw Dataset

`<a id="obtain-raw-dataset"></a>`

We follow the following methods to obtain raw data. We sincerely thank previous works.

| Dataset   | Preprocessing                                                                    |
| --------- | -------------------------------------------------------------------------------- |
| Few-NERD  | [Few-NERD](https://github.com/thunlp/Few-NERD)                                      |
| NYT       | [JointER](https://github.com/yubowen-ph/JointER/tree/master/dataset/NYT-multi/data) |
| ACE05-Evt | [OneIE](http://blender.cs.illinois.edu/software/oneie/)                             |

### Few-NERD

`<a id="few-nerd"></a>`

```bash
mkdir -p data/Few-NERD
wget -O data/Few-NERD/supervised.zip https://cloud.tsinghua.edu.cn/f/09265750ae6340429827/?dl=1
unzip -o -d data/Few-NERD/ data/Few-NERD/supervised.zip && rm data/Few-NERD/supervised.zip
python scripts/process_nerd.py
rm -r data/Few-NERD/supervised

$ tree data/Few-NERD 
data/Few-NERD
├── dev.txt
├── test.txt
└── train.txt

```

### NYT

`<a id="nyt"></a>`

```bash
mkdir data/NYT-multi
wget -P data/NYT-multi https://raw.githubusercontent.com/yubowen-ph/JointER/master/dataset/NYT-multi/data/train.json
wget -P data/NYT-multi https://raw.githubusercontent.com/yubowen-ph/JointER/master/dataset/NYT-multi/data/dev.json
wget -P data/NYT-multi https://raw.githubusercontent.com/yubowen-ph/JointER/master/dataset/NYT-multi/data/test.json

$ tree data/NYT-multi
data/Few-NERD
├── dev.json
├── test.json
└── train.json

```

### ACE05-Evt

`<a id="ace05-evt"></a>`

The preprocessing code of ACE05-Evt is following [OneIE](http://blender.cs.illinois.edu/software/oneie/).
Please follow the instructions and put preprocessed dataset at `data/oneie`:

```shell
## OneIE Preprocessing, ACE_DATA_FOLDER -> ace_2005_td_v7

$ tree data/oneie/ace05-EN 
data/oneie/ace05-EN
├── dev.oneie.json
├── english.json
├── english.oneie.json
├── test.oneie.json
└── train.oneie.json

```

Note:

- `nltk==3.5` is used in our experiments, we found `nltk==3.6+` may lead to different sentence numbers.
- Ensure that the tree structure of the data is consistent with what we have listed above.

# Convert Raw Dataset to Iteration Dataset

`<a id="convert-raw-dataset-to-iteration-datset"></a>`

You can obtain all iteration datasets by running the following code:

```shell
bash scripts/run_data_convert.bash
```

All iteration datasets will be located in the **`/AdaKGC/data`** directory.

## Details of Dataset Conversion

`<a id="details-of-dataset-conversion"></a>`

Now let's introduce the details of dataset conversion.

Each iteration dataset has a corresponding configuration file and the configuration file is placed under the directory **`AdaKGC/config/data_config`**.

#### Example of Dataset Configuration file

`<a id="example-of-dataset-configuration-file"></a>`

```yaml
# AdaKGC/config/data_config/event/ace05_event_M3.yaml
name: ace05_event
path: data/oneie/ace05-EN
data_class: OneIEEvent
split:
  train: train.oneie.json
  val: dev.oneie.json
  test: test.oneie.json
language: en
new_list:
 - acquit
 - appeal
 - business
 - contact
 - meet
 - merge organization
 - start organization
 - trial hearing
delete_list: 
  - Personnel:Elect 
  - Personnel:End-Position 
  - Personnel:Nominate 
  - Personnel:Start-Position
mapper:
  Business:Declare-Bankruptcy: business
  Business:End-Org: business
  Business:Merge-Org: merge organization
  Business:Start-Org: start organization
  Conflict:Attack: attack 
  Conflict:Demonstrate: demonstrate  
  Contact:Meet: meet 
  Contact:Phone-Write: contact
  Justice:Acquit: acquit
  Justice:Appeal: appeal
  Justice:Arrest-Jail: justice
  Justice:Charge-Indict: justice
  Justice:Convict: justice 
  Justice:Execute: execute 
  Justice:Extradite: extradite
  Justice:Fine: fine  
  Justice:Pardon: pardon
  Justice:Release-Parole: release parole 
  Justice:Sentence: sentence  
  Justice:Sue: sue  
  Justice:Trial-Hearing: trial hearing 
  Life:Be-Born: be born  
  Life:Die: die
  Life:Divorce: life 
  Life:Injure: life
  Life:Marry: marry
  Movement:Transport: transport
  Personnel:Elect: elect  
  Personnel:End-Position: end position
  Personnel:Nominate: nominate 
  Personnel:Start-Position: start position  
  Transaction:Transfer-Money: transfer money 
  Transaction:Transfer-Ownership: transfer ownership
  FAC: facility
  GPE: geographical social political
...
```

**`delete_list`** indicates the schema node to be deleted. For example, if a sample contains the **`Personal:Elect`** tag, but Personal:Elect is in delete_list, then delete the Personal:Elect tag of the sample.

**`new_list`** indicates the change of the schema of **`iteration n`** compared with **`iteration 1`** (i.e., the newly added schema node)

**`mapper`** is a **`label mapping`** between the raw dataset and the iteration dataset, which is important in vertical and replacement segmentation

## File Name Format

`<a id="file-name-format"></a>`

The configuration file name also indicates some information about the iteration dataset. E.g. **`ace05_event_H2`**, where **`H`** indicates that its segmentation mode is horizontal segmentation, and **`2`** indicates iteration 2.

#### Schema Hierarchy

`<a id="schema-hierarchy"></a>`

The schemas of the three datasets used in AdaKGC all have a hierarchical structure (on the NYT dataset, we have built a hierarchical schema ourselves). Schema hierarchy usually has a two-layer structure. Here, for simplicity, we call it **`parent class`** and **`child class`**. The parent class can be regarded as a coarse-grained label, while the subclass is a fine-grained label.

The following is an example of schema hierarchy in three datasets:

```text
Justice:Appeal         # ace05_event: Justice is parent class, Appeal is child class
building-library       # Few-NERD: building is parent class, library is child class
/business/company      # NYT: business is parent class, company is child class
```

#### Segmentation Mode

`<a id="segmentation-mode"></a>`

* **`H`**: Horizontal segmentation, Using only child class as label, each iteration will add several child class schema nodes (add or remove labels from **`delete_list`**). So the number of labels per iteration is variable.
* **`V`**: Vertical segmentation, Using both parent class and child class as label, each iteration **`mapper`** will change several labels from parent class to child class (**`delete_list`** is always empty). So the number of labels in each iteration is unchanged, but the content of labels change from parent class to child class.
* **`M`**: Mixed segmentation, A mixture of H and V.
* **`R`**: Replacement segmentation, Using only child class as label, each iteration **`mapper`** will change several labels from child class to their synonyms (**`delete_list`** is always empty). So the number of labels in each iteration is unchanged, but the content of labels change from child class to their synonyms. Therefore, the number of schema nodes remains unchanged, but the content of the nodes will become their synonyms.

#### Iteration change

`<a id="iteration-change"></a>`

Each iteration will change a fixed number of schema nodes, 2 for NYT, 3 for ACE05_Event, and 6 for Few-NERD. **`iteration 1`** usually has the minimum number of schema nodes, and **`iteration 7`** has the maximum number of schema nodes.

# Data Format

`<a id="data-format"></a>`

We use the same data format as UIE, each iteration dataset has the following items:

```shell
$ tree data/iter_1/NYT_H
data/iter_1/NYT_H
├── schema.json
├── test.json
├── train.json
└── val.json
```

#### Example of instance

`<a id="example-of-instance-from-ace05-event"></a>`

```json
ACE2005_Event
{
    "text": "She would be the first foreign woman to die in the wave of kidnappings in Iraq .", 
    "tokens": ["She", "would", "be", "the", "first", "foreign", "woman", "to", "die", "in", "the", "wave", "of", "kidnappings", "in", "Iraq", "."], 
    "record": "<extra_id_0> <extra_id_0> die <extra_id_5> die <extra_id_0> victim <extra_id_5> woman <extra_id_1> <extra_id_0> place <extra_id_5> Iraq <extra_id_1> <extra_id_1> <extra_id_1>", 
    "entity": [], 
    "relation": [], 
    "event": [
        {"type": "die", "offset": [8], "text": "die", "args": [{"type": "victim", "offset": [6], "text": "woman"}, 
        {"type": "place", "offset": [15], "text": "Iraq"}]}
    ], 
    "spot": ["die"], 
    "asoc": ["victim", "place"], 
    "spot_asoc": [
        {"span": "die", "label": "die", "asoc": [["victim", "woman"], ["place", "Iraq"]]}
    ]
}


NYT
{
    "text": "Should Turkey face eastward , toward its Muslim neighbors , or westward , toward Europe ?", 
    "tokens": ["Should", "Turkey", "face", "eastward", ",", "toward", "its", "Muslim", "neighbors", ",", "or", "westward", ",", "toward", "Europe", "?"], 
    "record": "<extra_id_0> <extra_id_0> location <extra_id_5> Turkey <extra_id_1> <extra_id_0> location <extra_id_5> Europe <extra_id_0> contains <extra_id_5> Turkey <extra_id_1> <extra_id_1> <extra_id_1>", 
    "entity": [
        {"type": "location", "offset": [14], "text": "Europe"}, 
        {"type": "location", "offset": [1], "text": "Turkey"}
    ], 
    "relation": [
        {"type": "contains", "args": [{"type": "location", "offset": [14], "text": "Europe"}, 
        {"type": "location", "offset": [1], "text": "Turkey"}]}
    ], 
    "event": [], 
    "spot": ["location"], 
    "asoc": ["contains"], 
    "spot_asoc": [
        {"span": "Turkey", "label": "location", "asoc": []}, 
        {"span": "Europe", "label": "location", "asoc": [["contains", "Turkey"]]}
    ]
}


Few-NERD
{
    "text": "Now Multan is the name of the city in Pakistan .", 
    "tokens": ["Now", "Multan", "is", "the", "name", "of", "the", "city", "in", "Pakistan", "."], 
    "record": "<extra_id_0> <extra_id_0> geographical social political <extra_id_5> Multan <extra_id_1> <extra_id_0> geographical social political <extra_id_5> Pakistan <extra_id_1> <extra_id_1>", 
    "entity": [
        {"type": "geographical social political", "offset": [9], "text": "Pakistan"}, 
        {"type": "geographical social political", "offset": [1], "text": "Multan"}
    ], 
    "relation": [], 
    "event": [], 
    "spot": ["geographical social political"], 
    "asoc": [], 
    "spot_asoc": [
        {"span": "Multan", "label": "geographical social political", "asoc": []}, 
        {"span": "Pakistan", "label": "geographical social political", "asoc": []}
    ]
}
```

# Acknowledgement

`<a id="acknowledgement"></a>`

Parts of the code are modified from [UIE](https://github.com/universal-ie/UIE). We appreciate the authors for making their projects open-sourced.

## Seq2Seq with Pointer Network

This is a Seq2Seq model with Copy Mechanism and Coverage Mechanism.

Note: The project refers to [atulkum](https://github.com/atulkum/pointer_summarizer) and [laihuiyuan](https://github.com/laihuiyuan/pointer-generator)

Datasets:

* `dataset1`: [CNN/DailyMail](https://arxiv.org/abs/1602.06023)

> You can download the dataset from [here](https://github.com/JafferWilson/Process-Data-of-CNN-DailyMail), and then run `dataprocess/process.py` to process data.
>
> The downloaded files should be placed in the specified directory.
>
> It defaults to `datasources`, which can be modified in `utils/parser.py`.
>
> Alternatively, you can follow the steps from [here](https://github.com/abisee/cnn-dailymail), processing the data by yourself.

Models:

* `model1`: [Get To The Point: Summarization with Pointer-Generator Networks](https://arxiv.org/abs/1704.04368)

> Other related models:
> 
> [Pointer Networks](https://arxiv.org/abs/1506.03134)
> 
> [Pointing the Unknown Words](https://arxiv.org/abs/1603.08148)
> 
> [Incorporating Copying Mechanism in Sequence-to-Sequence Learning](https://arxiv.org/abs/1603.06393)


### Data Process

```shell
PYTHONPATH=. python dataprocess/process.py
```

### Unit Test

* for loader

```shell
# loader(basic)
PYTHONPATH=. python loaders/loader1.py
# loader for copy mechanism
PYTHONPATH=. python loaders/loader1.py --is_copy
```

* for module

```shell
# module(basic)
PYTHONPATH=. python modules/module1.py
# module with mechanism copy
PYTHONPATH=. python modules/module1.py --is_copy
# module with mechanism coverage
PYTHONPATH=. python modules/module1.py --is_coverage
# module with mechanism copy and coverage
PYTHONPATH=. python modules/module1.py --is_copy --is_coverage
```

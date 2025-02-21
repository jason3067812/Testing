
# limitations under the License.

import copy

import unittest
from datasets import load_dataset, load_metric
from intel_extension_for_transformers.transformers import (
    metrics,
    OptimizedModel,
)

from neural_compressor.config import (
    DistillationConfig,
    KnowledgeDistillationLossConfig,
)
from intel_extension_for_transformers.transformers.trainer import NLPTrainer
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    set_seed,
)

os.environ["WANDB_DISABLED"] = "true"


class TestDistillation(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        set_seed(42)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            'distilbert-base-uncased'
        )
        self.teacher_model = AutoModelForSequenceClassification.from_pretrained(
            'distilbert-base-uncased-finetuned-sst-2-english'
        )
        raw_datasets = load_dataset("glue", "sst2")["validation"]
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        def preprocess_function(examples):
            # Tokenize the texts
            args = (
                (examples['sentence'],)
            )
            result = tokenizer(*args, padding=True, max_length=64, truncation=True)
            return result
        raw_datasets = raw_datasets.map(
            preprocess_function, batched=True, load_from_cache_file=True
        )
        eval_dataset = raw_datasets.select(range(30))
        self.dataset = eval_dataset

    @classmethod
    def tearDownClass(self):
        shutil.rmtree('./tmp_trainer', ignore_errors=True)
        shutil.rmtree('./distilled_model', ignore_errors=True)

    def test_fx_model_distil(self):
        metric = load_metric("accuracy")
        def compute_metrics(p):
            preds = p.predictions
            preds = np.argmax(preds, axis=1)
            return metric.compute(predictions=preds, references=p.label_ids)
        origin_weight = copy.deepcopy(self.model.classifier.weight)

        self.trainer = NLPTrainer(
            model=copy.deepcopy(self.model),
            train_dataset=self.dataset,
            eval_dataset=self.dataset,
            compute_metrics=compute_metrics,
        )
        metric_ = metrics.Metric(name="eval_accuracy")
        self.trainer.metrics = metric_
        distillation_criterion_conf = KnowledgeDistillationLossConfig(loss_types=["CE", "KL"])
        distillation_conf = DistillationConfig(self.teacher_model, distillation_criterion_conf)
        distilled_model = self.trainer.distill(
            distillation_config=distillation_conf
        )
        # By default, model will be saved in tmp_trainer dir.
        self.trainer.save_model('./distilled_model')
        loaded_model = OptimizedModel.from_pretrained(
            './distilled_model',
        )
        distilled_weight = copy.deepcopy(distilled_model.classifier.weight)
        loaded_weight = copy.deepcopy(loaded_model.classifier.weight)
        # check distilled model
        self.assertTrue((distilled_weight != origin_weight).any())
        # check loaded model
        self.assertTrue((distilled_weight == loaded_weight).all())
        mlflow.end_run()

    def test_functional_distil(self):
        def eval_func(model):
            return 1

        def train_func(model):
            return model

        self.trainer = NLPTrainer(self.model)

        distillation_conf = DistillationConfig(teacher_model=self.teacher_model)
        self.trainer.distill(distillation_conf,
                           provider="inc",
                           train_func = train_func,
                           eval_func = eval_func,)


if __name__ == "__main__":
    unittest.main()

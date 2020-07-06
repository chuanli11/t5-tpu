import functools
import os
import time
import sys
import argparse

import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds

import t5

import gzip
import json

# Improve logging.
from contextlib import contextmanager
import logging as py_logging


parser = argparse.ArgumentParser()
parser.add_argument('--model_size', type=str, default='samll',
                     help='model size')
parser.add_argument('--finetune_steps', type=int, default=100000,
                     help='number of steps for finetune')
parser.add_argument('--save_per_steps', type=int, default=100000,
                     help='number of steps for save model')
parser.add_argument('--base_dir', type=str, default='gs://caramel-spot-280923',
                     help='Google storage bucket')
parser.add_argument('--tpu_type', type=str, default='v2-8',
                     help='Google TPU instance type')
args = parser.parse_args()

def main():
    #MODEL_SIZE = "3B" #@param["small", "base", "large", "3B", "11B"]
    #FINETUNE_STEPS = 500 #@param {type: "integer"}
    #SAVE_PER_STEPS = 500
    MODEL_SIZE = args.model_size
    FINETUNE_STEPS = args.finetune_steps
    SAVE_PER_STEPS = args.save_per_steps
    BASE_DIR = args.base_dir
    if not BASE_DIR or BASE_DIR == "gs://":
      raise ValueError("You must enter a BASE_DIR.")
    DATA_DIR = os.path.join(BASE_DIR, "data")
    MODELS_DIR = os.path.join(BASE_DIR, "models-" + args.tpu_type)
    ON_CLOUD = True
    
    
    if ON_CLOUD:
      print("Setting up GCS access...")
      # Set credentials for GCS reading/writing from Colab and TPU.
      TPU_TOPOLOGY = "2x2"
      try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
        TPU_ADDRESS = tpu.get_master()
        print('Running on TPU:', TPU_ADDRESS)
      except ValueError:
        raise BaseException('ERROR: Not connected to a TPU runtime; please see the previous cell in this notebook for instructions!')
      tf.config.experimental_connect_to_host(TPU_ADDRESS)
    
    tf.disable_v2_behavior()
    
    
    if ON_CLOUD:
      tf.get_logger().propagate = False
      py_logging.root.setLevel('INFO')
    
    @contextmanager
    def tf_verbosity_level(level):
      og_level = tf.logging.get_verbosity()
      tf.logging.set_verbosity(level)
      yield
      tf.logging.set_verbosity(og_level)
    
    
    # Public directory of Natural Questions data on GCS.
    NQ_JSONL_DIR = "gs://natural_questions/v1.0-simplified/"
    NQ_SPLIT_FNAMES = {
        "train": "simplified-nq-train.jsonl.gz",
        "validation": "nq-dev-all.jsonl.gz"
    }
    nq_counts_path = os.path.join(DATA_DIR, "nq-counts.json")
    nq_tsv_path = {
        "train": os.path.join(DATA_DIR, "nq-train.tsv"),
        "validation": os.path.join(DATA_DIR, "nq-validation.tsv")
    }
    
    def nq_jsonl_to_tsv(in_fname, out_fname):
    
      def extract_answer(tokens, span):
        """Reconstruct answer from token span and remove extra spaces."""
        start, end = span["start_token"], span["end_token"]  
        ans = " ".join(tokens[start:end])
        # Remove incorrect spacing around punctuation.
        ans = ans.replace(" ,", ",").replace(" .", ".").replace(" %", "%")
        ans = ans.replace(" - ", "-").replace(" : ", ":").replace(" / ", "/")
        ans = ans.replace("( ", "(").replace(" )", ")")
        ans = ans.replace("`` ", "\"").replace(" ''", "\"")
        ans = ans.replace(" 's", "'s").replace("s ' ", "s' ")
        return ans
    
      count = 0
      with tf.io.gfile.GFile(in_fname, "rb") as infile,\
           tf.io.gfile.GFile(out_fname, "w") as outfile:
        for line in gzip.open(infile):
          ex = json.loads(line)
          # Remove any examples with more than one answer.
          if len(ex['annotations'][0]['short_answers']) != 1:
            continue
          # Questions in NQ do not include a question mark.
          question = ex["question_text"] + "?"
          answer_span = ex['annotations'][0]['short_answers'][0]
          # Handle the two document formats in NQ (tokens or text).
          if "document_tokens" in ex:
            tokens = [t["token"] for t in ex["document_tokens"]]
          elif "document_text" in ex:
            tokens = ex["document_text"].split(" ")
          answer = extract_answer(tokens, answer_span)
          # Write this line as <question>\t<answer>
          outfile.write("%s\t%s\n" % (question, answer))
          count += 1
          tf.logging.log_every_n(
              tf.logging.INFO,
              "Wrote %d examples to %s." % (count, out_fname),
              1000)
        return count
    
    if tf.io.gfile.exists(nq_counts_path):
      # Used cached data and counts.
      tf.logging.info("Loading NQ from cache.")
      num_nq_examples = json.load(tf.io.gfile.GFile(nq_counts_path))
    else:
      # Create TSVs and get counts.
      tf.logging.info("Generating NQ TSVs.")
      num_nq_examples = {}
      for split, fname in NQ_SPLIT_FNAMES.items():
        num_nq_examples[split] = nq_jsonl_to_tsv(
            os.path.join(NQ_JSONL_DIR, fname), nq_tsv_path[split])
      json.dump(num_nq_examples, tf.io.gfile.GFile(nq_counts_path, "w"))
    
    
    def nq_dataset_fn(split, shuffle_files=False):
      # We only have one file for each split.
      del shuffle_files
    
      # Load lines from the text file as examples.
      ds = tf.data.TextLineDataset(nq_tsv_path[split])
      # Split each "<question>\t<answer>" example into (question, answer) tuple.
      ds = ds.map(
          functools.partial(tf.io.decode_csv, record_defaults=["", ""],
                            field_delim="\t", use_quote_delim=False),
          num_parallel_calls=tf.data.experimental.AUTOTUNE)
      # Map each tuple to a {"question": ... "answer": ...} dict.
      ds = ds.map(lambda *ex: dict(zip(["question", "answer"], ex)))
      return ds
    
    print("A few raw validation examples...")
    for ex in tfds.as_numpy(nq_dataset_fn("validation").take(5)):
      print(ex)
    
    def trivia_preprocessor(ds):
      def normalize_text(text):
        """Lowercase and remove quotes from a TensorFlow string."""
        text = tf.strings.lower(text)
        text = tf.strings.regex_replace(text,"'(.*)'", r"\1")
        return text
    
      def to_inputs_and_targets(ex):
        """Map {"question": ..., "answer": ...}->{"inputs": ..., "targets": ...}."""
        return {
            "inputs":
                 tf.strings.join(
                     ["trivia question: ", normalize_text(ex["question"])]),
            "targets": normalize_text(ex["answer"])
        }
      return ds.map(to_inputs_and_targets, 
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    t5.data.TaskRegistry.add(
        "nq_context_free",
        # Supply a function which returns a tf.data.Dataset.
        dataset_fn=nq_dataset_fn,
        splits=["train", "validation"],
        # Supply a function which preprocesses text from the tf.data.Dataset.
        text_preprocessor=[trivia_preprocessor],
        # Lowercase targets before computing metrics.
        postprocess_fn=t5.data.postprocessors.lower_text, 
        # We'll use accuracy as our evaluation metric.
        metric_fns=[t5.evaluation.metrics.accuracy],
        # Not required, but helps for mixing and auto-caching.
        num_input_examples=num_nq_examples
    )
    
    
    nq_task = t5.data.TaskRegistry.get("nq_context_free")
    ds = nq_task.get_dataset(split="validation", sequence_length={"inputs": 128, "targets": 32})
    print("A few preprocessed validation examples...")
    for ex in tfds.as_numpy(ds.take(5)):
      print(ex)
    
    ds = tfds.load(
        "trivia_qa/unfiltered.nocontext",
        data_dir=DATA_DIR,
        # Download data locally for preprocessing to avoid using GCS space.
        download_and_prepare_kwargs={"download_dir": "./downloads"})
    print("A few raw validation examples...")
    for ex in tfds.as_numpy(ds["validation"].take(2)):
      print(ex)
    
    def tiviaqa_extract_qa(ds):
      def exract_qa(ex):
        return {
            "question": ex["question"],
            "answer": ex["answer"]["value"]
        }
      return ds.map(exract_qa, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    t5.data.TaskRegistry.add(
        "triviaqa_context_free",
        # A TfdsTask takes in a TFDS name instead of a tf.data.Dataset function.
        t5.data.TfdsTask,
        tfds_name="trivia_qa/unfiltered.nocontext:1.1.0",
        tfds_data_dir=DATA_DIR,
        text_preprocessor=[tiviaqa_extract_qa, trivia_preprocessor],
        postprocess_fn=t5.data.postprocessors.lower_text,
        metric_fns=[t5.evaluation.metrics.accuracy]
    )
    
    # Load and print a few examples.
    triviaqa_task = t5.data.TaskRegistry.get("triviaqa_context_free")
    ds = triviaqa_task.get_dataset(split="validation", sequence_length={"inputs": 128, "targets": 32})
    print("A few preprocessed validation examples...")
    for ex in tfds.as_numpy(ds.take(3)):
      print(ex)
    
    t5.data.MixtureRegistry.remove("trivia_all")
    t5.data.MixtureRegistry.add(
        "trivia_all",
        ["nq_context_free", "triviaqa_context_free"],
         default_rate=1.0
    )
    
    
    # Public GCS path for T5 pre-trained model checkpoints
    #BASE_PRETRAINED_DIR = "gs://t5-data/pretrained_models"
    BASE_PRETRAINED_DIR = "gs://t5-data/pretrained_models/cbqa"
    PRETRAINED_DIR = os.path.join(BASE_PRETRAINED_DIR, MODEL_SIZE)
    MODEL_DIR = os.path.join(MODELS_DIR, MODEL_SIZE)
    
    if ON_CLOUD and MODEL_SIZE == "3B":
        tf.logging.warn(
                "The `3B` model is too large to use with the 5GB GCS free tier. "
                "Make sure you have at least 25GB on GCS before continuing.")
    elif ON_CLOUD and MODEL_SIZE == "11B":
        tf.logging.warn(
                "The `11B` parameter is too large to fine-tune on the `v2-8` TPU "
                "Make sure you have at least a `v3-8` TPU instance.")
    
    # Set parallelism and batch size to fit on v2-8 TPU (if possible).
    # Limit number of checkpoints to fit within 5GB (if possible).
    model_parallelism, train_batch_size, keep_checkpoint_max = {
            "small": (1, 256, 16),
            "t5.1.1.small_ssm": (1, 256, 16),
            "base": (2, 128, 8),
            "large": (8, 64, 4),
            "t5.1.1.xl_ssm": (8, 16, 1),
            "3B": (8, 16, 1),
            "t5.1.1.xxl_ssm": (8, 16, 1),
            "11B": (8, 16, 1)}[MODEL_SIZE]
    
    tf.io.gfile.makedirs(MODEL_DIR)
    # The models from our paper are based on the Mesh Tensorflow Transformer.
    model = t5.models.MtfModel(
        model_dir=MODEL_DIR,
        tpu=TPU_ADDRESS,
        tpu_topology=TPU_TOPOLOGY,
        model_parallelism=model_parallelism,
        batch_size=train_batch_size,
        sequence_length={"inputs": 128, "targets": 32},
        learning_rate_schedule=0.003,
        save_checkpoints_steps=SAVE_PER_STEPS,
        keep_checkpoint_max=keep_checkpoint_max if ON_CLOUD else None,
        iterations_per_loop=100,
    )
    
    
    model.finetune(
            mixture_or_task_name="trivia_all",
            pretrained_model_dir=PRETRAINED_DIR,
            finetune_steps=FINETUNE_STEPS)
    
    model.batch_size = train_batch_size * 4
    model.eval(
            mixture_or_task_name="trivia_all",
            checkpoint_steps="all"
    )

if __name__ == "__main__":
    main()

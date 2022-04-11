# Databricks notebook source
# MAGIC %sh
# MAGIC 
# MAGIC wget https://github.com/bazelbuild/bazelisk/releases/download/v1.11.0/bazelisk-linux-amd64 -O /usr/local/bin/bazelisk
# MAGIC chmod +x /usr/local/bin/bazelisk

# COMMAND ----------

# MAGIC %sh
# MAGIC 
# MAGIC apt-get update
# MAGIC apt-get install -y --no-install-recommends graphviz 

# COMMAND ----------

# MAGIC %sh rsync -rv --ignore-errors ../.. /tmp/mlx

# COMMAND ----------

# MAGIC %pip install /tmp/mlx

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

# Patch IPython's display.
import IPython.core.display as icd
icd.display = display

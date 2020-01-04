"""
TF-Slim grouped API. Please see README.md for details and usage.
"""
# pylint: disable=unused-import

# Collapse tf-slim into a single namespace.
from inception.slim import inception_model as inception
from inception.slim import losses
from inception.slim import ops
from inception.slim import scopes
from inception.slim import variables
from inception.slim.scopes import arg_scope

#!/usr/bin/env python3
"""Print the public key hex for an existing Ed25519 private key (no secrets leave the machine)."""
import sys

from hardonian_foundry import signing

print(signing.public_key_hex(sys.argv[1]))

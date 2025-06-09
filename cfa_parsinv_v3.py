#!/usr/bin/env python
# coding: utf-8

import os
import re
import json
from collections import defaultdict
from collections.abc import MutableMapping

# For progress bar or advanced HTML parsing if needed later
import tqdm
from bs4 import BeautifulSoup
import pandas as pd
import ebooklib
from ebooklib import epub


# === Step 1: Parse Book and Module Titles ===

def parse_books_and_modules(text: str):
    lines = text.strip().split("\n")
    result = defaultdict(list)
    current_book = None

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("Book"):
            current_book = line
            result[current_book] = []
        elif line.startswith("Learning Module") and current_book:
            result[current_book].append(line.strip("Learning Module ").replace(":", "."))

    return dict(result)


# === Step 2: Read Input Files ===

with open("book_titles.txt", "r") as f:
    text_input = f.read()

books_modules = parse_books_and_modules(text_input)

with open("cfa2025.json", "r") as file:
    book_json = eval(file.read())

book_values = {o: f"Book {i + 1}: {o}" for i, o in enumerate(book_json.keys())}
book_json = {book_values[k]: v for k, v in book_json.items()}


# === Step 3: Text Cleaning and Chapter Parsing ===

def clean_text(text):
    text = re.sub(r'[\d\.]', '', text)
    cleaned = re.sub(r'\s+', ' ', text)
    return cleaned.strip().replace("Book : ", "")

def get_chap_level(x):
    x = x.split(".")
    if x[0].isnumeric():
        level1 = int(x[0])
        if len(x) > 1 and x[1].isnumeric():
            level2 = int(x[1])
            if len(x) > 2 and x[2].isnumeric():
                level3 = int(x[2])
                return [level1, level2, level3]
            return [level1, level2]
        return [level1]
    return []


# === Step 4: Define the Tree Structure ===

class BookNode(MutableMapping):
    def __init__(self, name="", content=None, dad_k=""):
        self.name = name
        self.content = content
        self._children = {}
        self.k = 1
        self.dad_k = dad_k

    def __getitem__(self, key):
        return self._children[key]

    def __setitem__(self, key, value):
        self._children[key] = value if isinstance(value, BookNode) else BookNode(name=key, content=value)

    def __delitem__(self, key):
        del self._children[key]

    def __iter__(self):
        return iter(self._children)

    def __len__(self):
        return len(self._children)

    def add_child(self, key, name=None, content=None):
        node_name = name if name else key
        current_k = self.k
        self.k += 1
        key = node_name = f"{current_k}. {clean_text(node_name)}"
        node = BookNode(name=node_name, content=content)
        self._children[key] = node
        return node

    def to_dict(self, max_levels=2, _level=1):
        if _level > max_levels:
            return len(self)
        return {key: child.to_dict(max_levels, _level + 1) for key, child in self._children.items()}

    def __repr__(self):
        return json.dumps(self.to_dict(max_levels=2), indent=4)

    def __str__(self):
        lines = []

        def walk(node, depth):
            indent = "  " * depth
            lines.append(f"{indent}- {node.name}")
            for child in node._children.values():
                walk(child, depth + 1)

        walk(self, 0)
        return "\n".join(lines)

    def flatten_paths(self, levels=(3, 4), _prefix=None, _depth=1):
        _prefix = _prefix if _prefix else []
        paths = []
        for key, child in self._children.items():
            cur_path = _prefix + [key]
            if _depth in levels:
                paths.append(" -> ".join(cur_path))
            paths.extend(child.flatten_paths(levels, cur_path, _depth + 1))
        return paths


# === Step 5: Build Tree Structure from JSON ===

cfa_book = BookNode(name="CFA Book")

for book in book_json.keys():
    current_book = cfa_book.add_child(book)
    modules = list(books_modules[book])
    chapter_list = list(book_json[book].keys())
    chapter_content = list(book_json[book].values())

    module_i = 0
    previous_chap_level = 20
    for k, con in zip(chapter_list, chapter_content):
        chap_level = get_chap_level(k)
        if not chap_level:
            continue

        if (previous_chap_level > chap_level[0]) and (module_i < len(modules)):
            current_module = current_book.add_child(modules[module_i])
            module_i += 1

        previous_chap_level = chap_level[0]

        if len(chap_level) == 1:
            current_c1 = current_module.add_child(k, content=con)
        elif len(chap_level) == 2:
            current_c2 = current_c1.add_child(k, content=con)
        elif len(chap_level) == 3:
            current_c3 = current_c2.add_child(k, content=con)


# # === Step 6: Output Flattened Paths ===

# flat_sections = cfa_book.flatten_paths(levels=(3, 4))

# # Example output of first path
# print(flat_sections[0] if flat_sections else "No sections found.")

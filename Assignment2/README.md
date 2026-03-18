# Assignment 2

This folder contains the implementation of Boolean retrieval and compression on top of the inverted index built in Assignment 1.

## File
- `assignment2_search_engine.py`

## Description
The program loads the inverted index generated in Assignment 1 and supports:

- AND Boolean search
- OR Boolean search
- NOT Boolean search
- Dictionary-as-a-String compression
- gap encoding
- variable-byte encoding

The compressed index is compared with the original index in terms of memory usage and query performance.

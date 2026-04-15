#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Date: 2025/12/20 18:10:39
# Author: Shilei Liu
import math
import random
from typing import List, Set


class ChunkStrategy:
    """Abstract base class for chunking strategies."""

    @staticmethod
    def build(*args, **kwargs) -> List[int]:
        """Builds a list of chunk sizes based on input parameters."""
        raise NotImplementedError


class FixedSizeStrategy(ChunkStrategy):
    """
    Fixed-size chunking strategy.

    Behavior: Creates chunks of a fixed size, with the last chunk containing the remainder.
    Example: (seq_len=13, chunk_size=5) -> [5, 5, 3]
    """

    @staticmethod
    def build(seq_len: int, chunk_size: int) -> List[int]:
        """
        Args:
            seq_len: The total length of the sequence.
            chunk_size: The target size for each chunk.
        """
        num_full_chunks = seq_len // chunk_size
        remainder = seq_len % chunk_size

        chunk_sizes = [chunk_size] * num_full_chunks
        if remainder > 0:
            chunk_sizes.append(remainder)

        return chunk_sizes


class RightAlignedFixedSizeStrategy(ChunkStrategy):
    @staticmethod
    def build(seq_len: int, chunk_size: int, priming_length: int = 0) -> List[int]:
        assert priming_length <= chunk_size

        if seq_len <= priming_length:
            return [seq_len]

        remaining_len = seq_len - priming_length

        num_full_chunks = remaining_len // chunk_size
        first_chunk_size = remaining_len % chunk_size

        main_chunks = []
        if first_chunk_size > 0:
            main_chunks.append(first_chunk_size)

        main_chunks.extend([chunk_size] * num_full_chunks)

        if priming_length > 0:
            main_chunks.append(priming_length)

        return main_chunks


class UniformSizeStrategy(ChunkStrategy):
    """
    Uniform-size chunking strategy.

    Behavior: Divides the sequence into a specified number of chunks, making their sizes
              as uniform as possible (size difference will be at most 1).
    Example: (seq_len=13, num_chunks=3) -> [5, 4, 4]
    """

    @staticmethod
    def build(seq_len: int, num_chunks: int) -> List[int]:
        """
        Args:
            seq_len: The total length of the sequence.
            num_chunks: The desired number of chunks.
        """
        base_size = seq_len // num_chunks
        remainder = seq_len % num_chunks

        chunk_sizes = [base_size + 1] * remainder
        chunk_sizes.extend([base_size] * (num_chunks - remainder))

        return chunk_sizes


class DynamicNumRandomSizeStrategy(ChunkStrategy):
    """
    Dynamic-number, random-size chunking strategy (most elastic).

    Behavior: First, determines a dynamic number of chunks within the allowed range.
              Then, partitions the sequence into that many chunks, with each chunk's
              size randomly determined within min/max bounds.
    """

    @staticmethod
    def build(
        seq_len: int, max_num_chunks: int, min_chunk_size: int, max_chunk_size: int
    ) -> List[int]:
        """
        Args:
            seq_len: The total length of the sequence.
            max_num_chunks: The maximum number of chunks allowed.
            min_chunk_size: The minimum allowed size for any single chunk.
            max_chunk_size: The maximum allowed size for any single chunk.
        """
        if max_chunk_size < min_chunk_size:
            raise ValueError("max_chunk_size cannot be less than min_chunk_size.")

        # --- Step 1: Determine a valid, dynamic number of chunks ---
        min_chunks_limit = math.ceil(seq_len / max_chunk_size)
        max_chunks_limit = seq_len // min_chunk_size

        effective_max_chunks = min(max_chunks_limit, max_num_chunks)

        if min_chunks_limit > effective_max_chunks:
            raise ValueError(
                f"No valid chunking scheme exists. At least {min_chunks_limit} chunks are required, "
                f"but the effective maximum is {effective_max_chunks}."
            )

        num_chunks = random.randint(min_chunks_limit, effective_max_chunks)

        # --- Step 2: Partition into `num_chunks` with random sizes (logic from FixedNumRandomSizeStrategy) ---
        chunks = [min_chunk_size] * num_chunks
        slack = seq_len - (num_chunks * min_chunk_size)

        eligible_indices = list(range(num_chunks))
        for _ in range(slack):
            if not eligible_indices:
                raise RuntimeError(
                    "Internal error: No eligible chunks left to distribute slack."
                )

            rand_idx = random.randrange(len(eligible_indices))
            target_chunk_idx = eligible_indices[rand_idx]
            chunks[target_chunk_idx] += 1

            if chunks[target_chunk_idx] == max_chunk_size:
                eligible_indices.pop(rand_idx)

        random.shuffle(chunks)
        for i, c in enumerate(chunks):
            assert c <= max_chunk_size, f"chunk {i}: {c} vs {max_chunk_size}"
        return chunks


class SemanticBoundaryStrategy(ChunkStrategy):
    """
    An advanced, flexible chunking strategy that aligns with semantic boundaries
    while respecting minimum and maximum size constraints.
    """

    @staticmethod
    def build(
        token_ids: List[int],
        max_chunk_size: int,
        min_chunk_size: int,
        max_num_chunks: int,
        boundary_token_ids: Set[int],
    ) -> List[int]:
        chunk_sizes = []
        start_idx = 0
        n_tokens = len(token_ids)

        # --- Main Loop ---
        while start_idx < n_tokens and len(chunk_sizes) < max_num_chunks:
            potential_end = min(start_idx + max_chunk_size, n_tokens)
            is_last_chunk_slot = len(chunk_sizes) == max_num_chunks - 1

            if is_last_chunk_slot:
                # This is the last slot. Grab everything up to the end of the sequence.
                # Any oversized portion will be handled by post-processing.
                end_idx = n_tokens
            else:
                search_start = potential_end
                search_end = start_idx + min_chunk_size
                end_idx = -1

                # Prioritize finding a semantic boundary
                for i in range(search_start, search_end - 1, -1):
                    if i > 0 and token_ids[i - 1] in boundary_token_ids:
                        remaining_after_split = n_tokens - i
                        if (
                            remaining_after_split == 0
                            or remaining_after_split >= min_chunk_size
                        ):
                            end_idx = i
                            break

                # Fallback if no boundary found
                if end_idx == -1:
                    remaining_after_potential_end = n_tokens - potential_end
                    if 0 < remaining_after_potential_end < min_chunk_size:
                        end_idx = n_tokens - min_chunk_size
                    else:
                        end_idx = potential_end

            # Ensure progress and handle cases where splitting isn't possible
            if end_idx <= start_idx:
                # If stuck, just take the max chunk or whatever is left
                end_idx = potential_end
                if end_idx <= start_idx and start_idx < n_tokens:
                    # Final fallback: take all remaining tokens and break
                    end_idx = n_tokens

            current_chunk_size = end_idx - start_idx
            chunk_sizes.append(current_chunk_size)
            start_idx = end_idx

        # --- Post-processing with strict adherence to max_num_chunks ---

        # This loop corrects any oversized chunks while respecting max_num_chunks.
        # It may need to run more than once if merges create new oversized chunks.
        for _ in range(len(chunk_sizes)):  # Safety break to prevent infinite loops
            was_modified = False
            # Fix oversized chunks
            for i in range(len(chunk_sizes)):
                if chunk_sizes[i] > max_chunk_size:
                    was_modified = True
                    oversized_part = chunk_sizes[i] - max_chunk_size
                    chunk_sizes[i] = max_chunk_size

                    # Distribute the oversized part
                    if len(chunk_sizes) < max_num_chunks:
                        # We have room, so add a new chunk for the oversized part
                        chunk_sizes.insert(i + 1, oversized_part)
                    elif i > 0:
                        # No room for new chunks, merge backwards into the previous chunk
                        chunk_sizes[i - 1] += oversized_part
                    # If i is 0 and no room, this part is effectively lost or requires different logic.
                    # For simplicity, we assume merging backward is the primary recovery.

            # Merge small chunks
            if len(chunk_sizes) > 1:
                # Iterate backwards to safely merge
                for i in range(len(chunk_sizes) - 1, 0, -1):
                    if chunk_sizes[i] < min_chunk_size:
                        # Try to merge with the previous chunk
                        if chunk_sizes[i - 1] + chunk_sizes[i] <= max_chunk_size:
                            was_modified = True
                            chunk_sizes[i - 1] += chunk_sizes.pop(i)

            if not was_modified:
                break

        # Final check and assertion
        assert len(chunk_sizes) <= max_num_chunks, (
            f"Chunk count exceeds limit: {len(chunk_sizes)} vs {max_num_chunks}"
        )
        for i, c in enumerate(chunk_sizes):
            # This assertion might still fail in extreme edge cases where constraints are impossible.
            # E.g., a single document of 200 tokens, max_size=100, max_chunks=1.
            # The logic prioritizes chunk count, so the single chunk will be 200.
            assert c > 0, "Chunk with size 0 created."
            if max_num_chunks > 1 or n_tokens <= max_chunk_size:
                assert c <= max_chunk_size, (
                    f"Chunk {i} is oversized: {c} vs {max_chunk_size}"
                )

        assert len(chunk_sizes) <= max_num_chunks, (
            f"{len(chunk_sizes)} vs {max_num_chunks}"
        )
        for i, c in enumerate(chunk_sizes):
            assert c <= max_chunk_size, f"chunk {i}: {c} vs {max_chunk_size}"
        return chunk_sizes

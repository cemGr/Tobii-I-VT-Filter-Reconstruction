# ivt_filter/strategies/windowing.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np

IndexPair = Tuple[Optional[int], Optional[int]]


class WindowSelector(ABC):
	"""Abstract base class for window selection strategies."""

	@abstractmethod
	def select(
		self,
		idx: int,
		times: np.ndarray,
		valid: np.ndarray,
		half_window_ms: float,
	) -> IndexPair:
		...


class TimeSymmetricWindowSelector(WindowSelector):
	"""Classic Olsen time-based window (may include invalid samples)."""

	def select(
		self,
		idx: int,
		times: np.ndarray,
		valid: np.ndarray,
		half_window_ms: float,
	) -> IndexPair:
		if not bool(valid[idx]):
			return None, None

		n = len(times)
		t_center = float(times[idx])

		first_idx = idx
		j = idx - 1
		while j >= 0:
			if t_center - float(times[j]) > half_window_ms:
				break
			first_idx = j
			j -= 1

		last_idx = idx
		k = idx + 1
		while k < n:
			if float(times[k]) - t_center > half_window_ms:
				break
			last_idx = k
			k += 1

		if first_idx == last_idx:
			return None, None

		return first_idx, last_idx


class SampleSymmetricWindowSelector(WindowSelector):
	"""Time-bounded, sample-symmetric window that requires validity."""

	def select(
		self,
		idx: int,
		times: np.ndarray,
		valid: np.ndarray,
		half_window_ms: float,
	) -> IndexPair:
		if not bool(valid[idx]):
			return None, None

		n = len(times)
		t_center = float(times[idx])

		left_indices = []
		j = idx - 1
		while j >= 0:
			if not bool(valid[j]):
				break
			if t_center - float(times[j]) > half_window_ms:
				break
			left_indices.append(j)
			j -= 1

		right_indices = []
		k = idx + 1
		while k < n:
			if not bool(valid[k]):
				break
			if float(times[k]) - t_center > half_window_ms:
				break
			right_indices.append(k)
			k += 1

		if not left_indices or not right_indices:
			return None, None

		m = min(len(left_indices), len(right_indices))
		if m <= 0:
			return None, None

		first_idx = left_indices[m - 1]
		last_idx = right_indices[m - 1]

		if first_idx >= last_idx:
			return None, None

		return first_idx, last_idx


class FixedSampleSymmetricWindowSelector(WindowSelector):
	"""Fixed-length sample window; allows invalids but requires valid endpoints."""

	def __init__(self, half_size: int):
		if half_size < 1:
			raise ValueError("half_size for FixedSampleSymmetricWindowSelector must be >= 1.")
		self.half_size = int(half_size)

	def select(
		self,
		idx: int,
		times: np.ndarray,
		valid: np.ndarray,
		half_window_ms: float,
	) -> IndexPair:
		if not bool(valid[idx]):
			return None, None

		n = len(times)
		window_start = max(0, idx - self.half_size)
		window_end = min(n - 1, idx + self.half_size)

		first_idx = None
		for j in range(window_start, idx + 1):
			if bool(valid[j]):
				first_idx = j
				break

		last_idx = None
		for k in range(window_end, idx - 1, -1):
			if bool(valid[k]):
				last_idx = k
				break

		if first_idx is None or last_idx is None:
			return None, None
		if first_idx >= last_idx:
			return None, None
		return first_idx, last_idx


class AsymmetricNeighborWindowSelector(WindowSelector):
	"""Asymmetric 2-sample window with backward/forward fallback."""

	def select(
		self,
		idx: int,
		times: np.ndarray,
		valid: np.ndarray,
		half_window_ms: float,
	) -> IndexPair:
		if not bool(valid[idx]):
			return None, None

		n = len(times)
		if idx > 0 and bool(valid[idx - 1]):
			return idx - 1, idx
		if idx < n - 1 and bool(valid[idx + 1]):
			return idx, idx + 1
		return None, None


class TimeBasedShiftedValidWindowSelector(WindowSelector):
	"""Shift time-based window to find contiguous valid block containing anchor.
	
	Similar to ShiftedValidWindowSelector but uses time constraints instead of sample counts.
	Optimized strategy:
	1. Check nominal time window [idx-half_window_ms .. idx+half_window_ms]
	2. If invalid samples found, shift window boundaries to exclude invalids
	3. Keep total time span constant while shifting
	4. Fallback if no full valid window achievable
	"""

	def __init__(self, fallback_mode: str = "shrink"):
		if fallback_mode not in ("shrink", "unclassified"):
			raise ValueError("fallback_mode must be 'shrink' or 'unclassified'.")
		self.fallback_mode = fallback_mode
		self._fallback_selector = TimeSymmetricWindowSelector()

	def select(
		self,
		idx: int,
		times: np.ndarray,
		valid: np.ndarray,
		half_window_ms: float,
	) -> IndexPair:
		if not bool(valid[idx]):
			return None, None

		n = len(valid)
		t_idx = times[idx]
		t_min = t_idx - half_window_ms
		t_max = t_idx + half_window_ms

		# Find nominal time window boundaries
		nominal_start = None
		for i in range(idx, -1, -1):
			if times[i] >= t_min:
				nominal_start = i
			else:
				break
		if nominal_start is None:
			nominal_start = 0

		nominal_end = None
		for i in range(idx, n):
			if times[i] <= t_max:
				nominal_end = i
			else:
				break
		if nominal_end is None:
			nominal_end = n - 1

		# Check if nominal window is fully valid
		if nominal_start <= idx <= nominal_end:
			if valid[nominal_start:nominal_end + 1].all():
				return nominal_start, nominal_end

		# Try to shift window while maintaining time span
		total_time_span = 2 * half_window_ms

		# Strategy: Find first invalid in nominal window, then try to shift
		invalid_positions = []
		for i in range(nominal_start, nominal_end + 1):
			if not valid[i]:
				invalid_positions.append(i)

		if not invalid_positions:
			# No invalids, use nominal
			return nominal_start, nominal_end

		# Find nearest invalid to idx
		nearest_invalid = min(invalid_positions, key=lambda i: abs(i - idx))

		# Shift window to exclude nearest invalid
		if nearest_invalid < idx:
			# Invalid on left, shift window right
			# New start after invalid
			new_start = nearest_invalid + 1
			if new_start >= n or not valid[new_start]:
				# Fallback
				if self.fallback_mode == "shrink":
					return self._fallback_selector.select(idx, times, valid, half_window_ms)
				return None, None

			# Find end that gives us total_time_span
			t_new_start = times[new_start]
			t_target_end = t_new_start + total_time_span

			new_end = None
			for i in range(new_start, n):
				if times[i] <= t_target_end:
					new_end = i
				else:
					break
			if new_end is None:
				new_end = n - 1

			# Check if valid and contains idx
			if new_start <= idx <= new_end:
				if valid[new_start:new_end + 1].all():
					return new_start, new_end
		else:
			# Invalid on right, shift window left
			# New end before invalid
			new_end = nearest_invalid - 1
			if new_end < 0 or not valid[new_end]:
				# Fallback
				if self.fallback_mode == "shrink":
					return self._fallback_selector.select(idx, times, valid, half_window_ms)
				return None, None

			# Find start that gives us total_time_span
			t_new_end = times[new_end]
			t_target_start = t_new_end - total_time_span

			new_start = None
			for i in range(new_end, -1, -1):
				if times[i] >= t_target_start:
					new_start = i
				else:
					break
			if new_start is None:
				new_start = 0

			# Check if valid and contains idx
			if new_start <= idx <= new_end:
				if valid[new_start:new_end + 1].all():
					return new_start, new_end

		# Fallback
		if self.fallback_mode == "shrink":
			return self._fallback_selector.select(idx, times, valid, half_window_ms)
		return None, None


class ShiftedValidWindowSelector(WindowSelector):
	"""Shift fixed-length window to find contiguous valid block containing anchor.
	
	Optimized strategy:
	1. Check nominal window [idx-half_size .. idx+half_size]
	2. If invalid samples found, identify nearest invalid to idx
	3. Cut at invalid, fill missing samples from opposite side
	4. Fallback if no full valid window achievable
	"""

	def __init__(self, half_size: int, fallback_mode: str = "shrink"):
		if half_size < 1:
			raise ValueError("half_size for ShiftedValidWindowSelector must be >= 1.")
		if fallback_mode not in ("shrink", "unclassified"):
			raise ValueError("fallback_mode must be 'shrink' or 'unclassified'.")
		self.half_size = int(half_size)
		self.fallback_mode = fallback_mode
		self._fallback_selector = FixedSampleSymmetricWindowSelector(half_size)

	def select(
		self,
		idx: int,
		times: np.ndarray,
		valid: np.ndarray,
		half_window_ms: float,
	) -> IndexPair:
		if not bool(valid[idx]):
			return None, None

		n = len(valid)
		window_len = 2 * self.half_size + 1

		# Define nominal window
		nominal_start = max(0, idx - self.half_size)
		nominal_end = min(n - 1, idx + self.half_size)
		
		# Check if nominal window is fully valid
		if nominal_start <= idx <= nominal_end:
			window_valid = valid[nominal_start:nominal_end + 1]
			if window_valid.all():
				return nominal_start, nominal_end
		
		# Optimization: Find nearest invalid sample to idx
		invalid_indices = []
		for offset in range(-self.half_size, self.half_size + 1):
			pos = idx + offset
			if 0 <= pos < n and not valid[pos]:
				invalid_indices.append((abs(offset), offset, pos))
		
		if not invalid_indices:
			# No invalids in range, use nominal
			return nominal_start, nominal_end
		
		# Sort by distance to idx
		invalid_indices.sort(key=lambda x: (x[0], abs(x[1])))
		nearest_invalid_dist, nearest_invalid_offset, nearest_invalid_pos = invalid_indices[0]
		
		# Cut at nearest invalid and fill from opposite side
		if nearest_invalid_offset > 0:
			# Invalid on right side, cut right, extend left
			right_cut = nearest_invalid_pos
			# Keep samples [?, idx] and exclude [right_cut, ...]
			# Need window_len samples total
			# Current valid span: nominal_start to right_cut-1
			current_valid_end = right_cut - 1
			needed_samples = window_len
			
			# Try to build window ending at current_valid_end
			new_start = current_valid_end - needed_samples + 1
			new_start = max(0, new_start)
			new_end = min(n - 1, new_start + needed_samples - 1)
			
			# Check if this window contains idx and is fully valid
			if new_start <= idx <= new_end:
				if valid[new_start:new_end + 1].all():
					return new_start, new_end
		else:
			# Invalid on left side, cut left, extend right
			left_cut = nearest_invalid_pos
			# Keep samples [idx, ?] and exclude [..., left_cut]
			current_valid_start = left_cut + 1
			needed_samples = window_len
			
			# Try to build window starting at current_valid_start
			new_start = current_valid_start
			new_end = min(n - 1, new_start + needed_samples - 1)
			
			# Check if this window contains idx and is fully valid
			if new_start <= idx <= new_end:
				if valid[new_start:new_end + 1].all():
					return new_start, new_end
		
		# Optimization failed, try original shifting strategy
		min_start = max(0, idx - self.half_size)
		max_start = min(idx, n - window_len)

		candidates = []
		for start in range(min_start, max_start + 1):
			end = start + window_len - 1
			if end >= n:
				break
			if not (start <= idx <= end):
				continue
			if bool(valid[start:end + 1].all()):
				shift = abs(start - nominal_start)
				candidates.append((shift, start))

		if candidates:
			candidates.sort(key=lambda t: (t[0], t[1]))
			chosen_start = candidates[0][1]
			return chosen_start, chosen_start + window_len - 1

		# Fallback
		if self.fallback_mode == "shrink":
			return self._fallback_selector.select(idx, times, valid, half_window_ms)
		return None, None

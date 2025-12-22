# ivt_filter/windowing.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np

IndexPair = Tuple[Optional[int], Optional[int]]


class WindowSelector(ABC):
    """
    Abstrakte Basis fuer alle Fenster-Strategien.

    select() gibt ein (first_idx, last_idx) Paar zurueck.
    Wenn kein sinnvolles Fenster gefunden wird, (None, None).
    """

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
    """
    Klassisches Olsen-Zeitfenster:

    - Start bei idx
    - nach links/rechts erweitern, solange innerhalb der half_window_ms
    - Ergebnis ist [first_idx, last_idx] mit first_idx < idx < last_idx
    - Ungültige Samples werden zugelassen (Validität wird später geprüft)
    """

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

        # Nach links gehen (unabhängig von Validität)
        first_idx = idx
        j = idx - 1
        while j >= 0:
            if t_center - float(times[j]) > half_window_ms:
                break
            first_idx = j
            j -= 1

        # Nach rechts gehen (unabhängig von Validität)
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
    """
    Zeit-begrenztes, aber sample-symmetrisches Fenster:

    - Gueltige Samples innerhalb half_window_ms links/rechts sammeln
    - Nimm min(#links, #rechts) gueltige Samples
    - Fenster ist dann symmetrisch um idx (gleich viele links/rechts)
    """

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

        # Kandidaten links
        left_indices = []
        j = idx - 1
        while j >= 0:
            if not bool(valid[j]):
                break
            if t_center - float(times[j]) > half_window_ms:
                break
            left_indices.append(j)
            j -= 1

        # Kandidaten rechts
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
    """
    Reines Sample-Fenster (ohne Zeitbegrenzung):

    - half_size = Ziel-Anzahl gueltiger Samples links und rechts.
    - half_window_ms wird hier ignoriert (nur fuer API-Kompatibilitaet).

    Verhalten:
      * Zuerst: versuche ein symmetrisches Fenster:
          - max. half_size gueltige Samples links und rechts,
          - Invalids werden uebersprungen,
          - wenn auf beiden Seiten >= 1 gueltiges Sample gefunden wird:
                m = min(#links, #rechts)
                Fenster = [links[m-1], rechts[m-1]]
      * Falls nur eine Seite gueltige Samples hat:
          - baue ein einseitiges Fenster:
                - nur links:  [links[m-1], idx]
                - nur rechts: [idx, rechts[m-1]]
      * Falls gar keine gueltigen Nachbarn gefunden werden:
          -> (None, None)  (kein Velocity)
    """

    def __init__(self, half_size: int):
        if half_size < 1:
            raise ValueError("half_size for FixedSampleSymmetricWindowSelector must be >= 1.")
        self.half_size = int(half_size)

    def select(
        self,
        idx: int,
        times: np.ndarray,
        valid: np.ndarray,
        half_window_ms: float,  # wird hier bewusst ignoriert
    ) -> IndexPair:
        if not bool(valid[idx]):
            return None, None

        n = len(times)

        # Links gueltige Samples einsammeln (Invalids ueberspringen)
        left_indices = []
        j = idx - 1
        while j >= 0 and len(left_indices) < self.half_size:
            if not bool(valid[j]):
                j -= 1
                continue
            left_indices.append(j)
            j -= 1

        # Rechts gueltige Samples einsammeln (Invalids ueberspringen)
        right_indices = []
        k = idx + 1
        while k < n and len(right_indices) < self.half_size:
            if not bool(valid[k]):
                k += 1
                continue
            right_indices.append(k)
            k += 1

        # Wenn nirgendwo Nachbarn: kein Fenster moeglich
        if not left_indices and not right_indices:
            return None, None

        # Fall 1: auf beiden Seiten gibt es gueltige Samples -> symmetrisches Fenster
        if left_indices and right_indices:
            m = min(len(left_indices), len(right_indices))
            if m <= 0:
                return None, None

            first_idx = left_indices[m - 1]
            last_idx = right_indices[m - 1]

            if first_idx >= last_idx:
                return None, None

            return first_idx, last_idx

        # Fall 2: nur links gueltige Samples -> einseitiges Fenster [links, idx]
        if left_indices and not right_indices:
            m = min(len(left_indices), self.half_size)
            first_idx = left_indices[m - 1]
            last_idx = idx

            if first_idx >= last_idx:
                return None, None

            return first_idx, last_idx

        # Fall 3: nur rechts gueltige Samples -> einseitiges Fenster [idx, rechts]
        if right_indices and not left_indices:
            m = min(len(right_indices), self.half_size)
            first_idx = idx
            last_idx = right_indices[m - 1]

            if first_idx >= last_idx:
                return None, None

            return first_idx, last_idx

        return None, None

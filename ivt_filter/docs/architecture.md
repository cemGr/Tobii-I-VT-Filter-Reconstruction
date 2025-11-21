# IVT filter architecture

This library mirrors the seven-step pipeline described by Olsen's I-VT identification filter. Each step is represented by a dedicated stage class in `ivt_filter.stages` and executed sequentially by `IVTFilterEngine`.

## Pipeline overview

The pipeline is intentionally staged so each function in Olsen's Section 3.1 maps to one class. The default order constructed by `IVTFilterEngine` is:

1. **GapFillingStage** — interpolates short gaps in per-eye data.
2. **EyeSelectionStage** — combines binocular gaze according to `EyeSelectionMode`.
3. **NoiseReductionStage** — smooths gaze using a strategy object (none, moving average, median).
4. **VelocityComputationStage** — derives angular velocity over a centered temporal window.
5. **IVTClassificationStage** — applies the velocity threshold to label samples and assemble gaze events.
6. **FixationMergingStage** — merges adjacent fixations separated by short spatial/temporal gaps.
7. **ShortFixationDiscardStage** — removes fixations shorter than the minimum duration and relabels their samples as unknown.

### Class-level view of the pipeline

```mermaid
classDiagram
    class IVTFilterEngine {
        +stages: list[IFilterStage]
        +run(recording, config) FilterResult
    }
    class IFilterStage {
        <<interface>>
        +process(recording, config)
    }
    class GapFillingStage
    class EyeSelectionStage
    class NoiseReductionStage
    class VelocityComputationStage
    class IVTClassificationStage
    class FixationMergingStage
    class ShortFixationDiscardStage

    class INoiseFilterStrategy {
        <<interface>>
        +apply(samples) list[Sample]
    }
    class NoNoiseFilterStrategy
    class MovingAverageNoiseFilterStrategy
    class MedianNoiseFilterStrategy

    IVTFilterEngine --> IFilterStage : orchestrates
    IFilterStage <|-- GapFillingStage
    IFilterStage <|-- EyeSelectionStage
    IFilterStage <|-- NoiseReductionStage
    IFilterStage <|-- VelocityComputationStage
    IFilterStage <|-- IVTClassificationStage
    IFilterStage <|-- FixationMergingStage
    IFilterStage <|-- ShortFixationDiscardStage

    NoiseReductionStage --> INoiseFilterStrategy : uses
    INoiseFilterStrategy <|-- NoNoiseFilterStrategy
    INoiseFilterStrategy <|-- MovingAverageNoiseFilterStrategy
    INoiseFilterStrategy <|-- MedianNoiseFilterStrategy
```

## Domain and event model

The domain model captures Tobii eye-tracking samples with explicit gaze and eye-position values for both eyes. It cleanly separates the raw input data from algorithmic stages.

```mermaid
classDiagram
    class EyeTrackingDataSet {
        +name: str
        +description: str?
        +sampling_rate_hz: float
        +source_path: str?
        +recordings: list[Recording]
    }
    class Recording {
        +id: str
        +start_time: datetime
        +end_time: datetime
        +samples: list[Sample]
        +dataset: EyeTrackingDataSet?
    }
    class Sample {
        +timestamp: datetime
        +left_validity: int
        +right_validity: int
        +left_eye: EyeData
        +right_eye: EyeData
        +combined_gaze_x: float?
        +combined_gaze_y: float?
        +angular_velocity_deg_per_sec: float?
        +label: str?
    }
    class EyeData {
        +gaze_x: float
        +gaze_y: float
        +eye_pos_x_3d: float?
        +eye_pos_y_3d: float?
        +eye_pos_z_3d: float?
    }

    class GazeEvent {
        <<abstract>>
        +start_time: datetime
        +end_time: datetime
        +duration: timedelta
        +event_type: GazeEventType
        +samples: list[Sample]
    }
    class Fixation {
        +position_x: float
        +position_y: float
        +sample_count: int
    }
    class Saccade {
        +peak_velocity: float
        +amplitude_deg: float
    }
    class UnknownSegment

    EyeTrackingDataSet "1" o-- "many" Recording
    Recording "1" o-- "many" Sample
    Sample "1" o-- "1" EyeData : left_eye
    Sample "1" o-- "1" EyeData : right_eye
    GazeEvent <|-- Fixation
    GazeEvent <|-- Saccade
    GazeEvent <|-- UnknownSegment
```

## Stage responsibilities mapped to Olsen

- **GapFillingStage** — Implements gap fill-in (Olsen §3.1.1) by interpolating short invalid stretches per eye.
- **EyeSelectionStage** — Implements eye selection (Olsen §3.1.2) with LEFT, RIGHT, AVERAGE, and STRICT_AVERAGE modes.
- **NoiseReductionStage** — Implements noise reduction (Olsen §3.1.3) using a configurable filter strategy.
- **VelocityComputationStage** — Implements velocity calculation (Olsen §3.1.4) via angular distance over a centered window.
- **IVTClassificationStage** — Implements I-VT classification (Olsen §3.1.5) using the velocity threshold to label fixations and saccades.
- **FixationMergingStage** — Implements merging adjacent fixations (Olsen §3.1.6) when spatial/temporal gaps are small.
- **ShortFixationDiscardStage** — Implements discarding short fixations (Olsen §3.1.7) and relabeling their samples as unknown.

## Strategy pattern for noise

Noise reduction is abstracted behind `INoiseFilterStrategy`, allowing the pipeline to remain closed for modification but open for extension: new filters can be added without touching the stages themselves. The default is a pass-through `NoNoiseFilterStrategy`, with moving average and median filters provided as ready-made strategies.

## SOLID alignment

- **SRP**: each stage handles exactly one responsibility from Olsen's paper.
- **OCP/DIP**: the engine depends on the `IFilterStage` abstraction and the noise stage depends on `INoiseFilterStrategy`, enabling alternative implementations via dependency injection.
- **LSP/ISP**: protocols and abstract base classes define the minimal contracts that substitutions must honor.

## Usage notes

1. Construct an `IVTFilterConfiguration` with the desired eye-selection mode, velocity threshold, gap and fixation thresholds, and a noise strategy.
2. Load Tobii TSV data into a `Recording` (via the test helpers or your own loader) and pass it to `IVTFilterEngine.run(recording, config)`.
3. Consume the returned `FilterResult.events` to access fixations, saccades, and unknown segments along with timing and positional data suitable for further analytics or visualization.

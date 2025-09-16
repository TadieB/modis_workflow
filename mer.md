```mermaid
graph TD
    subgraph "Setup & Configuration"
        A[<fa:fa-file-code /> config.yaml] --> B{<fa:fa-server /> HPC Orchestration};
        B -->|1. Submit Job| C[<fa:fa-terminal /> run_production_pipeline.sh];
        C -->|2. Setup Dask Cluster| D(<fa:fa-sitemap /> Dask Master/Workers on SLURM);
    end

    subgraph "Data Flow"
        E( <fa:fa-database /> LP DAAC <br> Raw HDF Tiles) --> F{<fa:fa-cogs /> Stage 1: Parallel Tile Processing};
        F -->|For each tile| G[<fa:fa-check-square /> QA & Preprocessing];
        G --> H[<fa:fa-cloud /> Cloud Mask Generation];
        H --> I[<fa:fa-layer-group /> Concatenate Bands <br> (7 Reflectance + 1 Cloud Mask)];
        I --> J([<fa:fa-hdd /> Disk: Processed Tiles <br> *.nc]);

        J --> K{<fa:fa-th-large /> Stage 2: Parallel Mosaicking};
        K -->|For each day| L[<fa:fa-clone /> Combine Processed Tiles];
        L --> M([<fa:fa-hdd /> Disk: Daily Mosaics <br> *.nc]);

        M --> N{{<fa:fa-hourglass-half /> Synchronization Barrier}};
        N -->|Wait for all mosaics| O{<fa:fa-th /> Stage 3: Parallel Patch Generation};
        O -->|Spatial Partitioning| P[<fa:fa-search /> Extract Patches & Label];
        P --> Q[<fa:fa-tags /> Create (Input, Target) Pairs];
        Q --> R{<fa:fa-folder-open /> Final Dataset};
    end

    subgraph "Output Formats"
        R --> S([<fa:fa-file-archive /> NetCDF Patches]);
        R --> T([<fa:fa-file-archive /> Zarr Store]);
    end

    %% Styling
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style B fill:#ccf,stroke:#333,stroke-width:2px
    style C fill:#ccf,stroke:#333,stroke-width:2px
    style D fill:#ccf,stroke:#333,stroke-width:2px
    style E fill:#ffe,stroke:#333,stroke-width:2px
    style J fill:#ffe,stroke:#333,stroke-width:2px,stroke-dasharray: 5 5
    style M fill:#ffe,stroke:#333,stroke-width:2px,stroke-dasharray: 5 5
    style S fill:#cfc,stroke:#333,stroke-width:2px
    style T fill:#cfc,stroke:#333,stroke-width:2px
    style N fill:#f99,stroke:#333,stroke-width:4px

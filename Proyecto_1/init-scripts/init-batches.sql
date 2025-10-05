-- Tabla para almacenar los batches de datos
CREATE TABLE IF NOT EXISTS data_batches (
    id SERIAL PRIMARY KEY,
    batch_number INTEGER NOT NULL,
    group_number INTEGER NOT NULL DEFAULT 5,
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    data JSONB NOT NULL,
    row_count INTEGER NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Índices para optimizar consultas
CREATE INDEX IF NOT EXISTS idx_batch_number ON data_batches(batch_number);
CREATE INDEX IF NOT EXISTS idx_timestamp ON data_batches(timestamp);
CREATE INDEX IF NOT EXISTS idx_batch_group ON data_batches(batch_number, group_number);

-- Tabla para rastrear el estado de los batches
CREATE TABLE IF NOT EXISTS batch_status (
    id SERIAL PRIMARY KEY,
    batch_number INTEGER UNIQUE NOT NULL,
    total_records INTEGER NOT NULL DEFAULT 0,
    last_updated TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    is_complete BOOLEAN NOT NULL DEFAULT FALSE
);

-- Tabla para almacenar el dataset acumulado para entrenamiento
CREATE TABLE IF NOT EXISTS training_dataset (
    id SERIAL PRIMARY KEY,
    batch_numbers INTEGER[] NOT NULL,
    total_records INTEGER NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    data JSONB NOT NULL
);

-- Vista para obtener estadísticas de batches
CREATE OR REPLACE VIEW batch_statistics AS
SELECT 
    batch_number,
    COUNT(*) as request_count,
    SUM(row_count) as total_rows,
    MIN(timestamp) as first_request,
    MAX(timestamp) as last_request,
    MAX(created_at) as batch_completion
FROM data_batches
GROUP BY batch_number
ORDER BY batch_number;

-- Comentarios en las tablas
COMMENT ON TABLE data_batches IS 'Almacena cada porción de datos recibida de la API externa';
COMMENT ON TABLE batch_status IS 'Rastrea el estado de completitud de cada batch';
COMMENT ON TABLE training_dataset IS 'Almacena datasets acumulados para entrenamiento de modelos';
COMMENT ON VIEW batch_statistics IS 'Vista con estadísticas agregadas por batch';

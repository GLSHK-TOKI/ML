export interface PIICategory {
    name: string;
    definition?: string | null;
}

export interface PIIDetectionConfig {
    categories: PIICategory[];
}
# HERDS (Higher Education Research and Development Survey) taxonomy.
#
# Based on NSF's HERDS discipline classification. Used by
# services/herds_classification_service.py as the target label set for
# proposal classification.
#
# Each subcategory holds a keyword list used for the rule-based
# component of the 0.75/0.25 semantic-vs-keyword blended score. The
# lists are deliberately non-overlapping — every term appears in
# exactly one category, chosen as the single strongest fit. Terms that
# are genuinely cross-domain were assigned to their root discipline
# (e.g. "thermodynamics" → Physics as the foundational science rather
# than also listing it under Chemistry or Mechanical Engineering).
#
# The topic_names entries are the underscore-joined strings used
# downstream in HERDS_FIELD output, aligned with the analytics DB's
# existing format for that column.

taxonomy = {

    # ──────────────────────────────────────────────────────────────
    # SCIENCE
    # ──────────────────────────────────────────────────────────────
    "Science": {
        "subcategories": {

            "Computer and Information Sciences": {
                "topic_names": ["Computer_and_Information_Sciences"],
                "keywords": [
                    "software", "programming", "algorithms", "data", "machine learning",
                    "artificial intelligence", "databases", "cloud computing",
                    "cybersecurity", "networks", "distributed systems",
                    "computer vision", "natural language processing",
                    "human computer interaction", "data science",
                    "high performance computing", "simulation"
                    # "optimization" moved to Mathematics and Statistics
                ]
            },

            "Biological and Biomedical Sciences": {
                "topic_names": ["Biological_Biomedical_Sciences"],
                "keywords": [
                    "biology", "genetics", "genomics", "molecular biology",
                    "cell biology", "immunology", "biochemistry",
                    "neuroscience", "biomedical", "disease", "microbiology",
                    "evolution", "bioinformatics"
                    # "ecosystems" moved to Natural Resources and Conservation
                ]
            },

            "Health Sciences": {
                "topic_names": ["Health_Sciences"],
                "keywords": [
                    "healthcare", "medicine", "clinical", "public health",
                    "epidemiology", "patient care", "medical treatment",
                    "diagnostics", "health policy", "biostatistics"
                ]
            },

            "Agricultural Sciences": {
                "topic_names": ["Agricultural_Sciences"],
                "keywords": [
                    "agriculture", "farming", "crop", "soil", "livestock",
                    "food production", "agronomy", "irrigation",
                    "agricultural systems", "food security"
                ]
            },

            "Natural Resources and Conservation": {
                "topic_names": ["Natural_Resources_Conservation"],
                "keywords": [
                    "conservation", "ecosystems", "wildlife", "biodiversity",
                    "natural resources", "environmental management",
                    "climate adaptation", "land use", "sustainability"
                ]
            },

            "Physics": {
                "topic_names": ["Physics"],
                "keywords": [
                    "quantum", "mechanics", "energy", "particle",
                    "thermodynamics", "electromagnetism", "relativity",
                    "plasma", "nuclear", "wave", "optics"
                ]
            },

            "Chemistry": {
                "topic_names": ["Chemistry"],
                "keywords": [
                    "chemical", "reactions", "synthesis", "molecules",
                    "organic", "inorganic", "analytical chemistry",
                    "spectroscopy", "compounds"
                    # "thermodynamics" moved to Physics
                ]
            },

            "Astronomy and Astrophysics": {
                "topic_names": ["Astronomy_and_Astrophysics"],
                "keywords": [
                    "cosmic", "stars", "galaxy", "black holes",
                    "cosmology", "space", "astrophysics",
                    "planetary", "radiation", "dark matter"
                ]
            },

            "Materials Science": {
                "topic_names": ["Materials_Science"],
                "keywords": [
                    "materials", "polymers", "nanomaterials",
                    "composites", "semiconductors",
                    "material properties", "fabrication"
                ]
            },

            "Mathematics and Statistics": {
                "topic_names": ["Mathematics_and_Statistics"],
                "keywords": [
                    "mathematics", "statistics", "probability",
                    "modeling", "optimization", "linear algebra",
                    "calculus", "statistical inference"
                    # "optimization" placed here as the foundational home
                ]
            },

            "Economics": {
                "topic_names": ["Economics"],
                "keywords": [
                    "economics", "markets", "finance", "trade",
                    "economic policy", "labor economics",
                    "macroeconomics", "microeconomics"
                    # "finance" placed here rather than Business/Management
                ]
            },

            "Political Science and Government": {
                "topic_names": ["Political_Science_and_Government"],
                "keywords": [
                    "policy", "government", "politics",
                    "public administration", "governance",
                    "elections", "legislation"
                    # "policy" placed here rather than Law
                ]
            },

            "Sociology and Demography": {
                "topic_names": ["Sociology_Demography_Population_Studies"],
                "keywords": [
                    "population", "demographics", "society",
                    "social systems", "inequality", "migration"
                ]
            },

            "Anthropology": {
                "topic_names": ["Anthropology"],
                "keywords": [
                    "culture", "archaeology", "ethnography",
                    "human evolution", "social behavior"
                    # "culture" placed here rather than Humanities
                ]
            },

            "Psychology": {
                "topic_names": ["Psychology"],
                "keywords": [
                    "behavior", "cognition", "mental health",
                    "psychological", "therapy", "personality"
                ]
            }
        }
    },

    # ──────────────────────────────────────────────────────────────
    # ENGINEERING
    # ──────────────────────────────────────────────────────────────
    "Engineering": {
        "subcategories": {

            "Civil Engineering": {
                "topic_names": ["Civil_Engineering"],
                "keywords": [
                    "infrastructure", "construction", "transportation",
                    "water systems", "structural engineering"
                ]
            },

            "Mechanical Engineering": {
                "topic_names": ["Mechanical_Engineering"],
                "keywords": [
                    "heat transfer", "fluid dynamics", "mechanical systems"
                    # "mechanics" and "thermodynamics" moved to Physics
                ]
            },

            "Electrical Engineering": {
                "topic_names": ["Electrical_Electronic_Communications_Engineering"],
                "keywords": [
                    "electronics", "signals", "communication",
                    "power systems", "circuits"
                ]
            },

            "Industrial Engineering": {
                "topic_names": ["Industrial_Manufacturing_Engineering"],
                "keywords": [
                    "manufacturing", "supply chain", "operations",
                    "logistics"
                    # "optimization" moved to Mathematics and Statistics
                    # "operations" placed here rather than Business/Management
                ]
            },

            "Chemical Engineering": {
                "topic_names": ["Chemical_Engineering"],
                "keywords": [
                    "chemical processes", "reactors", "process engineering",
                    "refining", "industrial chemistry"
                ]
            },

            "Aerospace Engineering": {
                "topic_names": ["Aerospace_Aeronautical_Astronautical_Engineering"],
                "keywords": [
                    "aerospace", "flight", "propulsion",
                    "space systems", "aerodynamics"
                ]
            }
        }
    },

    # ──────────────────────────────────────────────────────────────
    # NON-S&E
    # ──────────────────────────────────────────────────────────────
    "Non-S&E": {
        "subcategories": {

            "Business and Management": {
                "topic_names": ["Business_Management_Administration"],
                "keywords": [
                    "business", "management", "leadership",
                    "strategy"
                    # "operations" moved to Industrial Engineering
                    # "finance" moved to Economics
                ]
            },

            "Education": {
                "topic_names": ["Education"],
                "keywords": [
                    "education", "teaching", "curriculum",
                    "learning", "instruction"
                ]
            },

            "Law": {
                "topic_names": ["Law"],
                "keywords": [
                    "law", "legal", "regulation",
                    "litigation"
                    # "policy" moved to Political Science and Government
                ]
            },

            "Humanities": {
                "topic_names": ["Humanities"],
                "keywords": [
                    "history", "philosophy", "literature",
                    "arts"
                    # "culture" moved to Anthropology
                ]
            },

            "Social Work": {
                "topic_names": ["Social_Work"],
                "keywords": [
                    "social services", "counseling",
                    "community support", "welfare"
                ]
            }
        }
    }
}

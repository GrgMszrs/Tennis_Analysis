#!/usr/bin/env python3
"""
Tournament Name Normalization
Maps PBP tournament names to ATP standard names based on joinability analysis.
"""

import re

import pandas as pd

# Tournament mapping based on joinability analysis and common patterns
TOURNAMENT_MAPPING = {
    # Grand Slams
    "Gentlemen'sWimbledonSingles": "Wimbledon",
    "Men'sAustralianOpen": "Australian Open",
    "Men'sUSOpen": "US Open",
    "Men'sFrenchOpen": "French Open",
    "TheWimbledonLawnTennisChampionships": "Wimbledon",
    # Masters 1000
    "BNPParibasOpen-ATPIndianWells": "Indian Wells Masters",
    "BNPParibasMasters-ATPParis": "Paris Masters",
    "SonyOpenTennis-ATPMiami": "Miami Open",
    "MutualMadridOpen-ATPMadrid": "Madrid Open",
    "ItalianOpen-ATPRome": "Rome Masters",
    "Rogers&amp;Cupprés.ParYamaha-ATPMontreal": "Canadian Open",
    "Western&amp;SouthernOpen-ATPCincinnati": "Cincinnati Masters",
    "ShanghaiMasters-ATPShanghai": "Shanghai Masters",
    # ATP 500
    "BarcelonaOpenBancSabadell-ATPBarcelona": "Barcelona Open",
    "ATPSMercedesCup-ATPStuttgart": "Stuttgart Open",
    "MercedesCup-ATPStuttgart": "Stuttgart Open",
    "IfStockholmOpen-ATPStockholm": "Stockholm Open",
    "ABNAMROWorldTennisTourn.-ATPRotterdam": "Rotterdam Open",
    "CitiOpen-ATPWashington": "Washington Open",
    "BrisbanInternational-ATPBrisbane": "Brisbane International",
    # ATP 250
    "Winston-SalemOpen-ATPWinston-Salem": "Winston-Salem Open",
    "SkiStarSwedishOpen-ATPBastad": "Swedish Open",
    "ATPGermanyTennisChampionships-ATPHamburg": "Hamburg Open",
    "Generali&amp;OpenKitzbühel-ATPKitzbuhel": "Austrian Open",
    "J&amp;Safra&amp;Sarasin&amp;SwissIndoorsBasel-ATPBasel": "Swiss Indoors",
    # Other tournaments
    "DavisCup": "Davis Cup",
    "ATPWorldTourMasters1000&amp;BNPParibasOpen-ATPIndianWells": "Indian Wells Masters",
}


def normalize_tournament_name(tournament_name: str) -> str:
    """
    Normalize a tournament name to match ATP standards.

    Args:
        tournament_name: Raw tournament name from PBP data

    Returns:
        Normalized tournament name
    """
    if not tournament_name or str(tournament_name).lower() in ["nan", "none"]:
        return ""

    # Clean the name
    cleaned = str(tournament_name).strip()

    # Direct mapping
    if cleaned in TOURNAMENT_MAPPING:
        return TOURNAMENT_MAPPING[cleaned]

    # URL-based tournament name extraction
    if cleaned.startswith("http") or ".html" in cleaned:
        # Extract tournament name from URL patterns
        patterns = [
            r"atp([a-z]+)",  # e.g., atpbarcelona -> barcelona
            r"-atp([a-z]+)",  # e.g., tournament-atpwimbledon -> wimbledon
            r"([a-z]+)open",  # e.g., frenchopen -> french
            r"([a-z]+)masters",  # e.g., parismasters -> paris
        ]

        for pattern in patterns:
            match = re.search(pattern, cleaned.lower())
            if match:
                extracted = match.group(1).capitalize()
                return f"{extracted} Open" if "open" in pattern else f"{extracted} Masters"

    # Clean common patterns
    cleaned = re.sub(r"&amp;", "&", cleaned)  # HTML entities
    cleaned = re.sub(r"-ATP[A-Za-z]+", "", cleaned)  # Remove ATP codes
    cleaned = re.sub(r"\.html?", "", cleaned)  # Remove HTML extensions
    cleaned = re.sub(r"^https?://[^/]+/", "", cleaned)  # Remove URL prefixes

    # Specific pattern matching
    if "wimbledon" in cleaned.lower():
        return "Wimbledon"
    elif "australian" in cleaned.lower():
        return "Australian Open"
    elif "french" in cleaned.lower() or "roland" in cleaned.lower():
        return "French Open"
    elif "us open" in cleaned.lower() or "usopen" in cleaned.lower():
        return "US Open"
    elif "indian wells" in cleaned.lower():
        return "Indian Wells Masters"
    elif "miami" in cleaned.lower():
        return "Miami Open"
    elif "madrid" in cleaned.lower():
        return "Madrid Open"
    elif "rome" in cleaned.lower() or "italian" in cleaned.lower():
        return "Rome Masters"
    elif "montreal" in cleaned.lower() or "canada" in cleaned.lower():
        return "Canadian Open"
    elif "cincinnati" in cleaned.lower():
        return "Cincinnati Masters"
    elif "shanghai" in cleaned.lower():
        return "Shanghai Masters"
    elif "paris" in cleaned.lower() and "masters" in cleaned.lower():
        return "Paris Masters"

    return cleaned


def create_tournament_mapping_table(pbp_data: pd.DataFrame, atp_data: pd.DataFrame) -> pd.DataFrame:
    """
    Create a mapping table of PBP tournament names to ATP tournament names.

    Args:
        pbp_data: PBP dataset with tournament names
        atp_data: ATP dataset with tournament names

    Returns:
        DataFrame with mapping suggestions
    """
    # Get unique tournament names
    pbp_tournaments = pbp_data["tournament_name"].dropna().unique()
    atp_tournaments = atp_data["tourney_name"].dropna().unique()

    # Create mapping suggestions
    mapping_results = []

    for pbp_tournament in pbp_tournaments:
        normalized = normalize_tournament_name(pbp_tournament)

        # Find best ATP match
        best_match = None
        best_score = 0

        for atp_tournament in atp_tournaments:
            # Simple similarity scoring
            atp_lower = atp_tournament.lower()
            norm_lower = normalized.lower()

            if norm_lower in atp_lower or atp_lower in norm_lower:
                score = min(len(norm_lower), len(atp_lower)) / max(len(norm_lower), len(atp_lower))
                if score > best_score:
                    best_score = score
                    best_match = atp_tournament

        mapping_results.append(
            {"pbp_tournament": pbp_tournament, "normalized": normalized, "suggested_atp_match": best_match, "confidence": best_score}
        )

    return pd.DataFrame(mapping_results)


def apply_tournament_normalization(pbp_data: pd.DataFrame) -> pd.DataFrame:
    """
    Apply tournament name normalization to PBP data.

    Args:
        pbp_data: PBP dataset

    Returns:
        PBP dataset with normalized tournament names
    """
    result = pbp_data.copy()
    result["tournament_name_normalized"] = result["tournament_name"].apply(normalize_tournament_name)
    return result


if __name__ == "__main__":
    # Test normalization
    test_tournaments = [
        "Gentlemen'sWimbledonSingles",
        "Men'sAustralianOpen",
        "BNPParibasOpen-ATPIndianWells",
        "BarcelonaOpenBancSabadell-ATPBarcelona",
    ]

    print("🎾 Tournament Name Normalization Test")
    print("=" * 50)

    for tournament in test_tournaments:
        normalized = normalize_tournament_name(tournament)
        print(f"{tournament:<40} → {normalized}")

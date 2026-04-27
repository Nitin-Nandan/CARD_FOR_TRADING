"""
NSE 500 Stock Universe — Stock Price Prediction Project
==========================================================

Data Coverage Target:
  Pre-COVID  : Jan 2017 - Dec 2019
  Post-COVID : Jan 2022 - Feb 2026

Selection Criteria (in order of priority):
  1. Listed on NSE BEFORE January 2017 → full data for both windows
  2. Listed Jan 2017 - Dec 2021 → partial pre-COVID data, but still useful
  3. No stocks that underwent TICKER-BREAKING corporate actions
     (demergers that changed the continuity of price history)

Known Exclusions & Reasons:
  - TATAMOTORS    : Demerger of CV business (TMLCV) effective 2025 — price
                    history continuity broken; parent ticker restructured
  - JIOFIN        : Listed Aug 2023; zero 2017-2019 data
  - PAYTM         : Listed Nov 2021; zero 2017-2019 data
  - ZOMATO        : Listed Jul 2021; limited data (now renamed ETERNAL)
  - NYKAA         : Listed Nov 2021; zero 2017-2019 data
  - LIC           : Listed May 2022; zero 2017-2019 data
  - DELHIVERY     : Listed May 2022; zero 2017-2019 data
  - POLICYBAZAAR  : Listed Nov 2021; zero 2017-2019 data
  - RVNL          : Listed Apr 2019; minimal 2017-2019 data
  - IREDA         : Listed Nov 2023; no historical data
  - SWIGGY        : Listed Nov 2024; no historical data
  - VODAIDEA      : Extreme corporate stress / near-delisting risk
  - HDFCAMC       : Listed Jul 2018 (partial pre-COVID only, kept)
  - MIDHANI       : Listed Mar 2018 (partial pre-COVID, kept)
  - IRFC          : Listed Jan 2021; only post-COVID data, kept (still useful)
  - MUTHOOTFIN    : Listed Apr 2011; kept (full history)
  - CHOLAFIN      : Listed long ago; kept (full history)
  - ABFRL         : Demerger of Madura Fashion (ABLBL) announced 2024;
                    parent ABFRL ticker still trades, kept
  - ITC           : ITC Hotels demerged Jan 2025; parent ITC still trades, kept
  - SIEMENS       : SEIL demerged Jun 2025; parent SIEMENS still trades, kept
  - VEDL          : Demerger of sub-businesses ongoing; base VEDL ticker kept
                    but flagged — use with caution post-2024

Each stock entry:
  Symbol      : Fyers API format  "NSE:TICKER-EQ"
  ISIN        : International Securities Identification Number
  Ticker      : NSE raw ticker
  Name        : Full company name
  Listing Date: NSE listing date (YYYY-MM-DD)
  Sector      : Broad sector
  Notes       : Any data-quality caveats

Total: 500 stocks
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class StockInfo:
    symbol: str          # Fyers format: "NSE:TICKER-EQ"
    isin: str            # e.g. "INE002A01018"
    ticker: str          # e.g. "RELIANCE"
    name: str            # Full company name
    listing_date: str    # "YYYY-MM-DD"
    sector: str
    notes: str = ""


# =============================================================================
# SECTOR 1 — BANKING (Large & Mid Cap, all pre-2017 listings)
# =============================================================================
BANKING = [
    StockInfo("NSE:HDFCBANK-EQ",    "INE040A01034", "HDFCBANK",    "HDFC Bank Ltd",                    "1995-11-08", "Banking"),
    StockInfo("NSE:ICICIBANK-EQ",   "INE090A01021", "ICICIBANK",   "ICICI Bank Ltd",                   "1997-09-17", "Banking"),
    StockInfo("NSE:SBIN-EQ",        "INE062A01020", "SBIN",        "State Bank of India",              "1995-03-01", "Banking"),
    StockInfo("NSE:KOTAKBANK-EQ",   "INE237A01028", "KOTAKBANK",   "Kotak Mahindra Bank Ltd",          "1995-03-01", "Banking"),
    StockInfo("NSE:AXISBANK-EQ",    "INE238A01034", "AXISBANK",    "Axis Bank Ltd",                    "1998-11-16", "Banking"),
    StockInfo("NSE:INDUSINDBK-EQ",  "INE095A01012", "INDUSINDBK",  "IndusInd Bank Ltd",                "1998-01-28", "Banking"),
    StockInfo("NSE:BANKBARODA-EQ",  "INE028A01039", "BANKBARODA",  "Bank of Baroda",                   "1997-02-19", "Banking"),
    StockInfo("NSE:PNB-EQ",         "INE160A01022", "PNB",         "Punjab National Bank",             "2002-04-02", "Banking"),
    StockInfo("NSE:CANBK-EQ",       "INE476A01022", "CANBK",       "Canara Bank",                      "2002-12-23", "Banking"),
    StockInfo("NSE:UNIONBANK-EQ",   "INE692A01016", "UNIONBANK",   "Union Bank of India",              "2002-09-24", "Banking"),
    StockInfo("NSE:IDFCFIRSTB-EQ",  "INE092T01019", "IDFCFIRSTB",  "IDFC First Bank Ltd",              "2015-11-06", "Banking"),
    StockInfo("NSE:FEDERALBNK-EQ",  "INE171A01029", "FEDERALBNK",  "Federal Bank Ltd",                 "1995-03-01", "Banking"),
    StockInfo("NSE:RBLBANK-EQ",     "INE976G01028", "RBLBANK",     "RBL Bank Ltd",                     "2016-08-25", "Banking"),
    StockInfo("NSE:YESBANK-EQ",     "INE528G01035", "YESBANK",     "Yes Bank Ltd",                     "2005-07-12", "Banking", "Post-2020 restructuring; volatile but tradeable"),
    StockInfo("NSE:KARURVYSYA-EQ",  "INE036D01028", "KARURVYSYA",  "Karur Vysya Bank Ltd",             "2000-07-14", "Banking"),
    StockInfo("NSE:CUB-EQ",         "INE491A01021", "CUB",         "City Union Bank Ltd",              "1998-09-09", "Banking"),
    StockInfo("NSE:DCBBANK-EQ",     "INE503A01015", "DCBBANK",     "DCB Bank Ltd",                     "2006-07-28", "Banking"),
    StockInfo("NSE:SOUTHBANK-EQ",   "INE683A01023", "SOUTHBANK",   "South Indian Bank Ltd",            "1998-10-15", "Banking"),
    StockInfo("NSE:BANDHANBNK-EQ",  "INE545U01014", "BANDHANBNK",  "Bandhan Bank Ltd",                 "2018-03-27", "Banking", "Listed Mar 2018; partial pre-COVID"),
    StockInfo("NSE:AUBANK-EQ",      "INE949L01017", "AUBANK",      "AU Small Finance Bank Ltd",        "2017-07-10", "Banking", "Listed Jul 2017; partial pre-COVID"),
    StockInfo("NSE:EQUITASBNK-EQ",  "INE063P01018", "EQUITASBNK",  "Equitas Small Finance Bank Ltd",   "2020-10-02", "Banking", "Listed Oct 2020; post-COVID data only"),
    StockInfo("NSE:UJJIVANSFB-EQ",  "INE334L01012", "UJJIVANSFB",  "Ujjivan Small Finance Bank Ltd",   "2019-12-12", "Banking", "Listed Dec 2019; minimal pre-COVID"),
    StockInfo("NSE:SURYODAY-EQ",    "INE428N01019", "SURYODAY",    "Suryoday Small Finance Bank",      "2021-03-26", "Banking", "Listed Mar 2021"),
]

# =============================================================================
# SECTOR 2 — NON-BANKING FINANCE (NBFCs)
# =============================================================================
NBFC = [
    StockInfo("NSE:BAJFINANCE-EQ",  "INE296A01024", "BAJFINANCE",  "Bajaj Finance Ltd",                "2010-04-01", "NBFC"),
    StockInfo("NSE:BAJAJFINSV-EQ",  "INE918I01026", "BAJAJFINSV",  "Bajaj Finserv Ltd",                "2008-05-26", "NBFC"),
    StockInfo("NSE:SHRIRAMFIN-EQ",  "INE721A01013", "SHRIRAMFIN",  "Shriram Finance Ltd",              "1996-12-20", "NBFC"),
    StockInfo("NSE:CHOLAFIN-EQ",    "INE121A01024", "CHOLAFIN",    "Cholamandalam Investment & Finance","1997-05-30", "NBFC"),
    StockInfo("NSE:MUTHOOTFIN-EQ",  "INE414G01012", "MUTHOOTFIN",  "Muthoot Finance Ltd",              "2011-04-06", "NBFC"),
    StockInfo("NSE:M&MFIN-EQ",      "INE774D01024", "M&MFIN",      "Mahindra & Mahindra Financial",    "2006-02-10", "NBFC"),
    StockInfo("NSE:LICHSGFIN-EQ",   "INE115A01026", "LICHSGFIN",   "LIC Housing Finance Ltd",          "1998-05-26", "NBFC"),
    StockInfo("NSE:PNBHOUSING-EQ",  "INE572E01012", "PNBHOUSING",  "PNB Housing Finance Ltd",          "2016-11-07", "NBFC"),
    StockInfo("NSE:CANFINHOME-EQ",  "INE477A01020", "CANFINHOME",  "Can Fin Homes Ltd",                "1995-03-01", "NBFC"),
    StockInfo("NSE:RECLTD-EQ",      "INE020B01018", "RECLTD",      "REC Ltd",                          "2008-03-12", "NBFC"),
    StockInfo("NSE:PFC-EQ",         "INE134E01011", "PFC",         "Power Finance Corporation Ltd",    "2007-02-23", "NBFC"),
    StockInfo("NSE:SUNDARMFIN-EQ",  "INE660A01013", "SUNDARMFIN",  "Sundaram Finance Ltd",             "1998-10-13", "NBFC"),
    StockInfo("NSE:MANAPPURAM-EQ",  "INE522D01027", "MANAPPURAM",  "Manappuram Finance Ltd",           "2014-07-28", "NBFC"),
    StockInfo("NSE:IIFL-EQ",        "INE530B01024", "IIFL",        "IIFL Finance Ltd",                 "2005-05-05", "NBFC"),
    StockInfo("NSE:ABCAPITAL-EQ",   "INE674K01013", "ABCAPITAL",   "Aditya Birla Capital Ltd",         "2017-09-01", "NBFC", "Listed Sep 2017; partial pre-COVID"),
    StockInfo("NSE:POONAWALLA-EQ",  "INE511C01022", "POONAWALLA",  "Poonawalla Fincorp Ltd",           "2004-04-05", "NBFC"),
    StockInfo("NSE:CREDITACC-EQ",   "INE741K01010", "CREDITACC",   "CreditAccess Grameen Ltd",         "2018-08-23", "NBFC", "Listed Aug 2018; partial pre-COVID"),
    StockInfo("NSE:UGROCAP-EQ",     "INE551W01018", "UGROCAP",     "Ugro Capital Ltd",                 "2018-08-22", "NBFC", "Listed 2018"),
]

# =============================================================================
# SECTOR 3 — INSURANCE
# =============================================================================
INSURANCE = [
    StockInfo("NSE:HDFCLIFE-EQ",    "INE795G01014", "HDFCLIFE",    "HDFC Life Insurance Co Ltd",       "2017-11-17", "Insurance", "Listed Nov 2017; partial pre-COVID"),
    StockInfo("NSE:SBILIFE-EQ",     "INE123W01016", "SBILIFE",     "SBI Life Insurance Co Ltd",        "2017-10-03", "Insurance", "Listed Oct 2017; partial pre-COVID"),
    StockInfo("NSE:ICICIPRULI-EQ",  "INE726G01019", "ICICIPRULI",  "ICICI Prudential Life Insurance",  "2016-09-29", "Insurance"),
    StockInfo("NSE:ICICIGI-EQ",     "INE765G01017", "ICICIGI",     "ICICI Lombard General Insurance",  "2017-09-27", "Insurance", "Listed Sep 2017; partial pre-COVID"),
    StockInfo("NSE:GICRE-EQ",       "INE481Y01014", "GICRE",       "General Insurance Corporation",    "2017-10-25", "Insurance", "Listed Oct 2017"),
    StockInfo("NSE:NIACL-EQ",       "INE413U01017", "NIACL",       "New India Assurance Co Ltd",       "2017-11-13", "Insurance", "Listed Nov 2017"),
    StockInfo("NSE:STARHEALTH-EQ",  "INE264C01024", "STARHEALTH",  "Star Health & Allied Insurance",   "2021-12-10", "Insurance", "Listed Dec 2021; minimal 2022 data"),
]

# =============================================================================
# SECTOR 4 — CAPITAL MARKETS & AMC
# =============================================================================
CAPITAL_MARKETS = [
    StockInfo("NSE:HDFCAMC-EQ",     "INE127D01025", "HDFCAMC",     "HDFC Asset Management Co Ltd",     "2018-08-06", "Capital Markets", "Listed Aug 2018"),
    StockInfo("NSE:NAM-INDIA-EQ",   "INE120A01034", "NAM-INDIA",   "Nippon Life India Asset Mgmt",     "2017-11-06", "Capital Markets", "Listed Nov 2017"),
    StockInfo("NSE:ANGELONE-EQ",    "INE732I01013", "ANGELONE",    "Angel One Ltd",                    "2020-10-05", "Capital Markets", "Listed Oct 2020"),
    StockInfo("NSE:ICICIBANK-EQ",   "INE090A01021", "ICICIBANK",   "ICICI Bank Ltd",                   "1997-09-17", "Capital Markets"),  # duplicate guard — removed below
    StockInfo("NSE:BSE-EQ",         "INE118H01025", "BSE",         "BSE Ltd",                          "2017-02-03", "Capital Markets", "Listed Feb 2017"),
    StockInfo("NSE:MCX-EQ",         "INE745G01035", "MCX",         "Multi Commodity Exchange of India","2012-02-09", "Capital Markets"),
    StockInfo("NSE:CDSL-EQ",        "INE736A01011", "CDSL",        "Central Depository Services Ltd",  "2017-06-30", "Capital Markets", "Listed Jun 2017"),
    StockInfo("NSE:CAMS-EQ",        "INE596I01012", "CAMS",        "Computer Age Management Services", "2020-09-01", "Capital Markets", "Listed Sep 2020"),
    StockInfo("NSE:360ONE-EQ",      "INE466L01020", "360ONE",      "360 One Wam Ltd",                  "2021-07-19", "Capital Markets", "Listed Jul 2021"),
    StockInfo("NSE:MOTILALOFS-EQ",  "INE338A01024", "MOTILALOFS",  "Motilal Oswal Financial Services", "2007-09-25", "Capital Markets"),
    StockInfo("NSE:IIFLSEC-EQ",     "INE890H01039", "IIFLSEC",     "IIFL Securities Ltd",              "2019-09-11", "Capital Markets"),
]

# =============================================================================
# SECTOR 5 — IT & TECHNOLOGY
# =============================================================================
IT = [
    StockInfo("NSE:TCS-EQ",         "INE467B01029", "TCS",         "Tata Consultancy Services Ltd",    "2004-08-25", "IT"),
    StockInfo("NSE:INFY-EQ",        "INE009A01021", "INFY",        "Infosys Ltd",                      "1995-02-08", "IT"),
    StockInfo("NSE:HCLTECH-EQ",     "INE860A01027", "HCLTECH",     "HCL Technologies Ltd",             "2000-01-12", "IT"),
    StockInfo("NSE:WIPRO-EQ",       "INE075A01022", "WIPRO",       "Wipro Ltd",                        "1995-01-13", "IT"),
    StockInfo("NSE:TECHM-EQ",       "INE669C01036", "TECHM",       "Tech Mahindra Ltd",                "2006-08-28", "IT"),
    StockInfo("NSE:LTIMINDTREE-EQ",        "INE214T01019", "LTIMINDTREE",        "LTIMindtree Ltd",                  "2016-12-23", "IT"),
    StockInfo("NSE:MPHASIS-EQ",     "INE356A01018", "MPHASIS",     "Mphasis Ltd",                      "2004-09-07", "IT"),
    StockInfo("NSE:COFORGE-EQ",     "INE591G01017", "COFORGE",     "Coforge Ltd",                      "2004-08-31", "IT"),
    StockInfo("NSE:PERSISTENT-EQ",  "INE262H01021", "PERSISTENT",  "Persistent Systems Ltd",           "2010-03-30", "IT"),
    StockInfo("NSE:OFSS-EQ",        "INE881D01027", "OFSS",        "Oracle Financial Services Software","2002-09-09", "IT"),
    StockInfo("NSE:HEXAWARE-EQ",    "INE093A01033", "HEXAWARE",    "Hexaware Technologies Ltd",        "2002-06-01", "IT", "Re-listed Feb 2025 after delisting in 2020; use post-2022 data only"),
    StockInfo("NSE:KPITTECH-EQ",    "INE058A01010", "KPITTECH",    "KPIT Technologies Ltd",            "2019-04-25", "IT", "Current KPIT entity listed Apr 2019 post-demerger"),
    StockInfo("NSE:TATAELXSI-EQ",   "INE670A01012", "TATAELXSI",   "Tata Elxsi Ltd",                   "2003-09-26", "IT"),
    StockInfo("NSE:LTTS-EQ",        "INE010V01017", "LTTS",        "L&T Technology Services Ltd",      "2016-09-23", "IT"),
    StockInfo("NSE:SONATSOFTW-EQ",  "INE269A01021", "SONATSOFTW",  "Sonata Software Ltd",              "1999-09-07", "IT"),
    StockInfo("NSE:MASTEK-EQ",      "INE759A01020", "MASTEK",      "Mastek Ltd",                       "1997-10-23", "IT"),
    StockInfo("NSE:CYIENT-EQ",      "INE136B01020", "CYIENT",      "Cyient Ltd",                       "2004-10-14", "IT"),
    # NIITTECH removed — same company as COFORGE (renamed); duplicate ISIN INE591G01017
    StockInfo("NSE:RATEGAIN-EQ",    "INE117M01021", "RATEGAIN",    "RateGain Travel Technologies",     "2021-12-17", "IT", "Listed Dec 2021"),
    StockInfo("NSE:NEWGEN-EQ",      "INE619B01017", "NEWGEN",      "Newgen Software Technologies",     "2018-01-29", "IT", "Listed Jan 2018"),
    StockInfo("NSE:INTELLECT-EQ",   "INE306R01017", "INTELLECT",   "Intellect Design Arena Ltd",       "2014-12-18", "IT"),
    StockInfo("NSE:ROUTE-EQ",       "INE419U01018", "ROUTE",       "Route Mobile Ltd",                 "2020-09-21", "IT", "Listed Sep 2020"),
    StockInfo("NSE:TANLA-EQ",       "INE483C01032", "TANLA",       "Tanla Platforms Ltd",              "2000-06-15", "IT"),
    StockInfo("NSE:HAPPSTMNDS-EQ",  "INE419U01012", "HAPPSTMNDS",  "Happiest Minds Technologies",      "2020-09-17", "IT", "Listed Sep 2020"),
    StockInfo("NSE:MSTCLTD-EQ",     "INE098F01031", "MSTCLTD",     "MSTC Ltd",                         "2019-03-29", "IT"),
]

# =============================================================================
# SECTOR 6 — ENERGY, OIL & GAS
# =============================================================================
ENERGY = [
    StockInfo("NSE:RELIANCE-EQ",    "INE002A01018", "RELIANCE",    "Reliance Industries Ltd",          "1995-11-29", "Energy"),
    StockInfo("NSE:ONGC-EQ",        "INE213A01029", "ONGC",        "Oil & Natural Gas Corporation",    "1995-03-01", "Energy"),
    StockInfo("NSE:BPCL-EQ",        "INE029A01011", "BPCL",        "Bharat Petroleum Corporation",     "1995-03-01", "Energy"),
    StockInfo("NSE:IOC-EQ",         "INE242A01010", "IOC",         "Indian Oil Corporation Ltd",       "1996-06-10", "Energy"),
    StockInfo("NSE:HINDPETRO-EQ",   "INE094A01015", "HINDPETRO",   "Hindustan Petroleum Corp Ltd",     "1995-03-01", "Energy"),
    StockInfo("NSE:GAIL-EQ",        "INE129A01019", "GAIL",        "GAIL (India) Ltd",                 "1997-04-02", "Energy"),
    StockInfo("NSE:PETRONET-EQ",    "INE347G01014", "PETRONET",    "Petronet LNG Ltd",                 "2004-03-26", "Energy"),
    StockInfo("NSE:IGL-EQ",         "INE203G01027", "IGL",         "Indraprastha Gas Ltd",             "2003-12-26", "Energy"),
    StockInfo("NSE:MGL-EQ",         "INE890G01024", "MGL",         "Mahanagar Gas Ltd",                "2016-07-01", "Energy"),
    StockInfo("NSE:GUJGASLTD-EQ",   "INE844O01030", "GUJGASLTD",   "Gujarat Gas Ltd",                  "2015-03-26", "Energy"),
    StockInfo("NSE:CASTROLIND-EQ",  "INE172A01027", "CASTROLIND",  "Castrol India Ltd",                "1998-09-14", "Energy"),
    StockInfo("NSE:OIL-EQ",         "INE274J01014", "OIL",         "Oil India Ltd",                    "2009-09-30", "Energy"),
    StockInfo("NSE:MRPL-EQ",        "INE103A01014", "MRPL",        "Mangalore Refinery & Petrochem",   "1994-09-15", "Energy"),
    StockInfo("NSE:CHENNPETRO-EQ",  "INE178A01016", "CHENNPETRO",  "Chennai Petroleum Corporation",    "1995-03-01", "Energy"),
]

# =============================================================================
# SECTOR 7 — POWER & UTILITIES
# =============================================================================
POWER = [
    StockInfo("NSE:NTPC-EQ",        "INE733E01010", "NTPC",        "NTPC Ltd",                         "2004-11-05", "Power"),
    StockInfo("NSE:POWERGRID-EQ",   "INE752E01010", "POWERGRID",   "Power Grid Corporation of India",  "2007-10-05", "Power"),
    StockInfo("NSE:COALINDIA-EQ",   "INE522F01014", "COALINDIA",   "Coal India Ltd",                   "2010-11-04", "Power"),
    StockInfo("NSE:ADANIGREEN-EQ",  "INE364U01010", "ADANIGREEN",  "Adani Green Energy Ltd",           "2018-06-18", "Power", "Listed Jun 2018"),
    StockInfo("NSE:ADANIPOWER-EQ",  "INE814H01011", "ADANIPOWER",  "Adani Power Ltd",                  "2009-08-20", "Power"),
    StockInfo("NSE:TATAPOWER-EQ",   "INE245A01021", "TATAPOWER",   "Tata Power Company Ltd",           "1995-03-01", "Power"),
    StockInfo("NSE:CESC-EQ",        "INE486A01013", "CESC",        "CESC Ltd",                         "1995-03-01", "Power"),
    StockInfo("NSE:NHPC-EQ",        "INE848E01016", "NHPC",        "NHPC Ltd",                         "2009-09-01", "Power"),
    StockInfo("NSE:SJVN-EQ",        "INE002L01015", "SJVN",        "SJVN Ltd",                         "2010-05-20", "Power"),
    StockInfo("NSE:TORNTPOWER-EQ",  "INE813H01021", "TORNTPOWER",  "Torrent Power Ltd",                "2007-05-28", "Power"),
    StockInfo("NSE:JSWENERGY-EQ",   "INE121E01018", "JSWENERGY",   "JSW Energy Ltd",                   "2010-01-04", "Power"),
    StockInfo("NSE:SUZLON-EQ",      "INE040H01021", "SUZLON",      "Suzlon Energy Ltd",                "2005-10-10", "Power"),
    StockInfo("NSE:INOXWIND-EQ",    "INE066P01011", "INOXWIND",    "Inox Wind Ltd",                    "2015-04-09", "Power"),
    StockInfo("NSE:KPIL-EQ",        "INE880J01026", "KPIL",        "Kalpataru Projects International", "2002-07-09", "Power"),
    StockInfo("NSE:GPPL-EQ",          "INE508L01018", "GPPL",          "Gujarat Pipavav Port Ltd",             "2009-08-26", "Infra",            "Pre-2017; full history; port infra"),
]

# =============================================================================
# SECTOR 8 — AUTOMOBILES
# =============================================================================
AUTO = [
    StockInfo("NSE:MARUTI-EQ",      "INE585B01010", "MARUTI",      "Maruti Suzuki India Ltd",          "2003-07-09", "Auto"),
    StockInfo("NSE:M&M-EQ",         "INE101A01026", "M&M",         "Mahindra & Mahindra Ltd",          "1995-03-01", "Auto"),
    StockInfo("NSE:BAJAJ-AUTO-EQ",  "INE917I01010", "BAJAJ-AUTO",  "Bajaj Auto Ltd",                   "2008-05-26", "Auto"),
    StockInfo("NSE:HEROMOTOCO-EQ",  "INE158A01026", "HEROMOTOCO",  "Hero MotoCorp Ltd",                "2002-07-01", "Auto"),
    StockInfo("NSE:EICHERMOT-EQ",   "INE066A01021", "EICHERMOT",   "Eicher Motors Ltd",                "2004-09-07", "Auto"),
    StockInfo("NSE:TVSMOTOR-EQ",    "INE494B01023", "TVSMOTOR",    "TVS Motor Company Ltd",            "2000-08-02", "Auto"),
    StockInfo("NSE:ASHOKLEY-EQ",    "INE208A01029", "ASHOKLEY",    "Ashok Leyland Ltd",                "1995-03-01", "Auto"),
    StockInfo("NSE:BOSCHLTD-EQ",    "INE323A01026", "BOSCHLTD",    "Bosch Ltd",                        "2004-05-10", "Auto"),
    StockInfo("NSE:MOTHERSON-EQ",   "INE775A01035", "MOTHERSON",   "Samvardhana Motherson Int Ltd",    "2004-07-27", "Auto"),
    StockInfo("NSE:BHARATFORG-EQ",  "INE465A01025", "BHARATFORG",  "Bharat Forge Ltd",                 "1995-03-01", "Auto"),
    StockInfo("NSE:APOLLOTYRE-EQ",  "INE438A01022", "APOLLOTYRE",  "Apollo Tyres Ltd",                 "1995-03-01", "Auto"),
    StockInfo("NSE:MRF-EQ",         "INE883A01011", "MRF",         "MRF Ltd",                          "1996-09-30", "Auto"),
    StockInfo("NSE:CEATLTD-EQ",     "INE482A01020", "CEATLTD",     "CEAT Ltd",                         "1995-03-01", "Auto"),
    StockInfo("NSE:BALKRISIND-EQ",  "INE787D01026", "BALKRISIND",  "Balkrishna Industries Ltd",        "2002-07-01", "Auto"),
    StockInfo("NSE:SUNDRMFAST-EQ",  "INE387A01021", "SUNDRMFAST",  "Sundram Fasteners Ltd",            "1995-03-01", "Auto"),
    StockInfo("NSE:EXIDEIND-EQ",    "INE302A01020", "EXIDEIND",    "Exide Industries Ltd",             "1995-03-01", "Auto"),
    StockInfo("NSE:AMARARAJA-EQ",  "INE885A01032", "AMARARAJA",  "Amara Raja Energy & Mobility Ltd",     "1995-03-01", "Auto"),
    StockInfo("NSE:UNOMINDA-EQ",    "INE247K01021", "UNOMINDA",    "Uno Minda Ltd",             "1997-12-22", "Auto"),
    StockInfo("NSE:SUPRAJIT-EQ",    "INE399C01030", "SUPRAJIT",    "Suprajit Engineering Ltd",         "1995-03-01", "Auto"),
    StockInfo("NSE:CRAFTSMAN-EQ",   "INE00X901017", "CRAFTSMAN",   "Craftsman Automation Ltd",         "2021-03-25", "Auto", "Listed Mar 2021"),
    StockInfo("NSE:OLECTRA-EQ",     "INE260J01025", "OLECTRA",     "Olectra Greentech Ltd",            "2000-05-08", "Auto"),
]

# =============================================================================
# SECTOR 9 — FMCG & CONSUMER
# =============================================================================
FMCG = [
    StockInfo("NSE:HINDUNILVR-EQ",  "INE030A01027", "HINDUNILVR",  "Hindustan Unilever Ltd",           "1995-03-01", "FMCG"),
    StockInfo("NSE:ITC-EQ",         "INE154A01025", "ITC",         "ITC Ltd",                          "1995-03-01", "FMCG", "Hotels demerged Jan 2025; parent ticker continues"),
    StockInfo("NSE:NESTLEIND-EQ",   "INE239A01016", "NESTLEIND",   "Nestle India Ltd",                 "1994-09-15", "FMCG"),
    StockInfo("NSE:BRITANNIA-EQ",   "INE216A01030", "BRITANNIA",   "Britannia Industries Ltd",         "1998-11-03", "FMCG"),
    StockInfo("NSE:DABUR-EQ",       "INE016A01026", "DABUR",       "Dabur India Ltd",                  "1999-04-28", "FMCG"),
    StockInfo("NSE:GODREJCP-EQ",    "INE102D01028", "GODREJCP",    "Godrej Consumer Products Ltd",     "2001-06-05", "FMCG"),
    StockInfo("NSE:MARICO-EQ",      "INE196A01026", "MARICO",      "Marico Ltd",                       "1996-05-01", "FMCG"),
    StockInfo("NSE:COLPAL-EQ",      "INE259A01022", "COLPAL",      "Colgate-Palmolive (India) Ltd",    "1995-03-01", "FMCG"),
    StockInfo("NSE:EMAMILTD-EQ",    "INE548C01032", "EMAMILTD",    "Emami Ltd",                        "2010-03-24", "FMCG"),
    StockInfo("NSE:TATACONSUM-EQ",  "INE192A01025", "TATACONSUM",  "Tata Consumer Products Ltd",       "1998-11-03", "FMCG"),
    StockInfo("NSE:VBL-EQ",    "INE200M01013", "VBL",    "Varun Beverages Ltd",              "2016-11-08", "FMCG"),
    StockInfo("NSE:RADICO-EQ",      "INE944F01028", "RADICO",      "Radico Khaitan Ltd",               "2003-06-16", "FMCG"),
    StockInfo("NSE:UBL-EQ",         "INE686F01025", "UBL",         "United Breweries Ltd",             "2001-08-20", "FMCG"),
    StockInfo("NSE:UNITDSPR-EQ",  "INE854D01024", "UNITDSPR",  "United Spirits Ltd",               "1996-07-11", "FMCG"),
    StockInfo("NSE:ZYDUSWELL-EQ",   "INE768C01010", "ZYDUSWELL",   "Zydus Wellness Ltd",               "2001-02-13", "FMCG"),
    StockInfo("NSE:JYOTHYLAB-EQ",   "INE668F01031", "JYOTHYLAB",   "Jyothy Labs Ltd",                  "2007-12-21", "FMCG"),
    StockInfo("NSE:PGHH-EQ",        "INE199A01012", "PGHH",        "Procter & Gamble Hygiene",         "1995-03-01", "FMCG"),
    StockInfo("NSE:GILLETTE-EQ",    "INE322A01010", "GILLETTE",    "Gillette India Ltd",               "1995-03-01", "FMCG"),
    # VBL removed — exact duplicate of VARUNBEV (same company, same ISIN INE200M01013)
    StockInfo("NSE:BIKAJI-EQ",      "INE00OI01017", "BIKAJI",      "Bikaji Foods International Ltd",   "2022-11-07", "FMCG", "Listed Nov 2022; post-COVID only"),
]

# =============================================================================
# SECTOR 10 — RETAIL & CONSUMPTION
# =============================================================================
RETAIL = [
    StockInfo("NSE:TITAN-EQ",       "INE280A01028", "TITAN",       "Titan Company Ltd",                "1995-03-01", "Retail"),
    StockInfo("NSE:ASIANPAINT-EQ",  "INE021A01026", "ASIANPAINT",  "Asian Paints Ltd",                 "1995-03-01", "Retail"),
    StockInfo("NSE:TRENT-EQ",       "INE849A01020", "TRENT",       "Trent Ltd",                        "1998-11-13", "Retail"),
    StockInfo("NSE:DMART-EQ",       "INE192R01011", "DMART",       "Avenue Supermarts Ltd",            "2017-03-21", "Retail", "Listed Mar 2017"),
    StockInfo("NSE:NAUKRI-EQ",      "INE663F01024", "NAUKRI",      "Info Edge (India) Ltd",            "2006-11-21", "Retail"),
    StockInfo("NSE:JUBLFOOD-EQ",    "INE797F01020", "JUBLFOOD",    "Jubilant Foodworks Ltd",           "2010-02-08", "Retail"),
    StockInfo("NSE:ABFRL-EQ",       "INE647O01011", "ABFRL",       "Aditya Birla Fashion & Retail",    "2015-05-13", "Retail", "Madura Fashion demerger approved; parent ABFRL continues"),
    StockInfo("NSE:PAGEIND-EQ",     "INE761H01022", "PAGEIND",     "Page Industries Ltd",              "2007-03-16", "Retail"),
    StockInfo("NSE:VSTIND-EQ",      "INE302B01027", "VSTIND",      "VST Industries Ltd",               "1995-03-01", "Retail"),
    StockInfo("NSE:INDIAMART-EQ",   "INE933S01016", "INDIAMART",   "IndiaMART InterMESH Ltd",          "2019-07-04", "Retail"),
    StockInfo("NSE:BATAINDIA-EQ",        "INE176A01028", "BATAINDIA",        "Bata India Ltd",                   "1995-03-01", "Retail"),
    StockInfo("NSE:RELAXO-EQ",      "INE131B01039", "RELAXO",      "Relaxo Footwears Ltd",             "1995-03-01", "Retail"),
    StockInfo("NSE:SHOPERSTOP-EQ",  "INE498B01024", "SHOPERSTOP",  "Shoppers Stop Ltd",                "2005-04-20", "Retail"),
    StockInfo("NSE:VMART-EQ",       "INE469I01019", "VMART",       "V-Mart Retail Ltd",                "2013-02-21", "Retail"),
    StockInfo("NSE:CAMPUS-EQ",      "INE00GY01011", "CAMPUS",      "Campus Activewear Ltd",            "2022-04-26", "Retail", "Listed Apr 2022"),
    StockInfo("NSE:PCBL-EQ",          "INE346H01014", "PCBL",          "PCBL Ltd (Phillips Carbon Black)",     "2017-08-07", "Chemicals",        "Demerged Aug 2017; active trading"),
]

# =============================================================================
# SECTOR 11 — PHARMA & HEALTHCARE
# =============================================================================
PHARMA = [
    StockInfo("NSE:SUNPHARMA-EQ",   "INE044A01036", "SUNPHARMA",   "Sun Pharmaceutical Industries",    "1994-09-15", "Pharma"),
    StockInfo("NSE:DRREDDY-EQ",     "INE089A01023", "DRREDDY",     "Dr. Reddy's Laboratories Ltd",     "1994-09-15", "Pharma"),
    StockInfo("NSE:CIPLA-EQ",       "INE059A01026", "CIPLA",       "Cipla Ltd",                        "1995-03-01", "Pharma"),
    StockInfo("NSE:DIVISLAB-EQ",    "INE361B01024", "DIVISLAB",    "Divi's Laboratories Ltd",          "2003-03-12", "Pharma"),
    StockInfo("NSE:AUROPHARMA-EQ",  "INE406A01037", "AUROPHARMA",  "Aurobindo Pharma Ltd",             "2000-07-19", "Pharma"),
    StockInfo("NSE:BIOCON-EQ",      "INE376G01013", "BIOCON",      "Biocon Ltd",                       "2004-04-07", "Pharma"),
    StockInfo("NSE:TORNTPHARM-EQ",  "INE685A01028", "TORNTPHARM",  "Torrent Pharmaceuticals Ltd",      "2002-01-22", "Pharma"),
    StockInfo("NSE:LUPIN-EQ",       "INE326A01037", "LUPIN",       "Lupin Ltd",                        "2001-09-10", "Pharma"),
    StockInfo("NSE:ALKEM-EQ",       "INE540L01014", "ALKEM",       "Alkem Laboratories Ltd",           "2015-12-23", "Pharma"),
    StockInfo("NSE:ABBOTINDIA-EQ",  "INE358A01014", "ABBOTINDIA",  "Abbott India Ltd",                 "1995-03-01", "Pharma"),
    StockInfo("NSE:GLENMARK-EQ",    "INE935A01035", "GLENMARK",    "Glenmark Pharmaceuticals Ltd",     "2000-02-07", "Pharma"),
    StockInfo("NSE:ZYDUSLIFE-EQ",    "INE010B01027", "ZYDUSLIFE",    "Zydus Lifesciences Ltd",           "2010-03-18", "Pharma"),
    StockInfo("NSE:IPCALAB-EQ",        "INE571A01020", "IPCALAB",        "IPCA Laboratories Ltd",            "2003-09-19", "Pharma"),
    StockInfo("NSE:LALPATHLAB-EQ",  "INE600L01024", "LALPATHLAB",  "Dr Lal PathLabs Ltd",              "2015-12-23", "Pharma"),
    StockInfo("NSE:METROPOLIS-EQ",  "INE592B01026", "METROPOLIS",  "Metropolis Healthcare Ltd",        "2019-04-15", "Pharma"),
    StockInfo("NSE:APOLLOHOSP-EQ",  "INE437A01024", "APOLLOHOSP",  "Apollo Hospitals Enterprise Ltd",  "1995-03-01", "Pharma"),
    StockInfo("NSE:MAXHEALTH-EQ",   "INE027H01010", "MAXHEALTH",   "Max Healthcare Institute Ltd",     "2020-08-21", "Pharma", "Listed Aug 2020"),
    StockInfo("NSE:FORTIS-EQ",      "INE061F01013", "FORTIS",      "Fortis Healthcare Ltd",            "2007-05-09", "Pharma"),
    StockInfo("NSE:NH-EQ",          "INE062M01020", "NH",          "Narayana Hrudayalaya Ltd",         "2016-01-06", "Pharma"),
    StockInfo("NSE:SUVEN-EQ",       "INE278A01037", "SUVEN",       "Suven Pharmaceuticals Ltd",        "2020-01-22", "Pharma", "Current entity demerged Jan 2020"),
    StockInfo("NSE:GRANULES-EQ",    "INE101D01020", "GRANULES",    "Granules India Ltd",               "2005-01-24", "Pharma"),
    StockInfo("NSE:SOLARA-EQ",      "INE707Q01026", "SOLARA",      "Solara Active Pharma Sciences",    "2018-10-01", "Pharma", "Listed 2018"),
    StockInfo("NSE:SYNGENE-EQ",     "INE398R01022", "SYNGENE",     "Syngene International Ltd",        "2015-08-11", "Pharma"),
    StockInfo("NSE:ALKYLAMINE-EQ",  "INE150B01021", "ALKYLAMINE",  "Alkyl Amines Chemicals Ltd",       "1995-03-01", "Pharma"),
    StockInfo("NSE:NATCOPHARM-EQ",  "INE987B01026", "NATCOPHARM",  "Natco Pharma Ltd",                 "1995-09-12", "Pharma"),
]

# =============================================================================
# SECTOR 12 — METALS & MINING
# =============================================================================
METALS = [
    StockInfo("NSE:TATASTEEL-EQ",   "INE081A01020", "TATASTEEL",   "Tata Steel Ltd",                   "1995-03-01", "Metals"),
    StockInfo("NSE:HINDALCO-EQ",    "INE038A01020", "HINDALCO",    "Hindalco Industries Ltd",          "1997-01-09", "Metals"),
    StockInfo("NSE:JSWSTEEL-EQ",    "INE019A01038", "JSWSTEEL",    "JSW Steel Ltd",                    "2004-03-11", "Metals"),
    StockInfo("NSE:SAIL-EQ",        "INE114A01011", "SAIL",        "Steel Authority of India Ltd",     "1995-03-01", "Metals"),
    StockInfo("NSE:VEDL-EQ",        "INE205A01025", "VEDL",        "Vedanta Ltd",                      "1998-05-13", "Metals", "Demerger of multiple units ongoing; VEDL parent ticker trades"),
    StockInfo("NSE:NATIONALUM-EQ",  "INE139A01034", "NATIONALUM",  "National Aluminium Co Ltd",        "1995-03-01", "Metals"),
    StockInfo("NSE:NMDC-EQ",        "INE584A01023", "NMDC",        "NMDC Ltd",                         "1995-03-01", "Metals"),
    StockInfo("NSE:MOIL-EQ",        "INE490G01020", "MOIL",        "MOIL Ltd",                         "2010-12-15", "Metals"),
    StockInfo("NSE:HINDCOPPER-EQ",  "INE531E01026", "HINDCOPPER",  "Hindustan Copper Ltd",             "2010-11-30", "Metals"),
    StockInfo("NSE:WELCORP-EQ",     "INE191B01025", "WELCORP",     "Welspun Corp Ltd",                 "1995-03-01", "Metals"),
    StockInfo("NSE:APLAPOLLO-EQ",         "INE702C01027", "APLAPOLLO",         "APL Apollo Tubes Ltd",             "2011-11-14", "Metals"),
    StockInfo("NSE:RATNAMANI-EQ",   "INE703B01027", "RATNAMANI",   "Ratnamani Metals & Tubes Ltd",     "2000-09-22", "Metals"),
    StockInfo("NSE:JINDALSAW-EQ",   "INE324A01016", "JINDALSAW",   "Jindal Saw Ltd",                   "1995-03-01", "Metals"),
    StockInfo("NSE:JSWHL-EQ",       "INE137O01025", "JSWHL",       "JSW Holdings Ltd",                 "2005-01-20", "Metals"),
    # APLAPOLLO removed — duplicate of APL (APL Apollo Tubes Ltd, INE702C01027)
]

# =============================================================================
# SECTOR 13 — CEMENT & CONSTRUCTION MATERIALS
# =============================================================================
CEMENT = [
    StockInfo("NSE:ULTRACEMCO-EQ",  "INE481G01011", "ULTRACEMCO",  "UltraTech Cement Ltd",             "2004-09-14", "Cement"),
    StockInfo("NSE:GRASIM-EQ",      "INE047A01021", "GRASIM",      "Grasim Industries Ltd",            "1995-03-01", "Cement"),
    StockInfo("NSE:AMBUJACEM-EQ",   "INE079A01024", "AMBUJACEM",   "Ambuja Cements Ltd",               "1995-03-01", "Cement"),
    StockInfo("NSE:ACC-EQ",         "INE012A01025", "ACC",         "ACC Ltd",                          "1995-03-01", "Cement"),
    StockInfo("NSE:SHREECEM-EQ",    "INE070A01015", "SHREECEM",    "Shree Cement Ltd",                 "1995-03-01", "Cement"),
    StockInfo("NSE:RAMCOCEM-EQ",    "INE331A01037", "RAMCOCEM",    "The Ramco Cements Ltd",            "1995-03-01", "Cement"),
    StockInfo("NSE:JKCEMENT-EQ",    "INE823G01014", "JKCEMENT",    "JK Cement Ltd",                    "2004-10-11", "Cement"),
    StockInfo("NSE:HEIDELBERG-EQ",  "INE578A01026", "HEIDELBERG",  "HeidelbergCement India Ltd",       "2003-12-29", "Cement"),
    StockInfo("NSE:ORIENTCEM-EQ",   "INE918G01018", "ORIENTCEM",   "Orient Cement Ltd",                "2012-02-16", "Cement"),
    StockInfo("NSE:NUVOCO-EQ",      "INE621H01010", "NUVOCO",      "Nuvoco Vistas Corp Ltd",           "2021-08-23", "Cement", "Listed Aug 2021"),
]

# =============================================================================
# SECTOR 14 — INFRASTRUCTURE & CONSTRUCTION
# =============================================================================
INFRA = [
    StockInfo("NSE:LT-EQ",          "INE018A01030", "LT",          "Larsen & Toubro Ltd",              "1995-03-01", "Infra"),
    StockInfo("NSE:NCC-EQ",         "INE868B01028", "NCC",         "NCC Ltd",                          "1995-03-01", "Infra"),
    StockInfo("NSE:HCC-EQ",         "INE549A01026", "HCC",         "Hindustan Construction Co Ltd",    "1995-03-01", "Infra"),
    StockInfo("NSE:AHLUCONT-EQ",    "INE758C01029", "AHLUCONT",    "Ahluwalia Contracts Ltd",          "2007-10-08", "Infra"),
    StockInfo("NSE:KNRCON-EQ",      "INE634I01029", "KNRCON",      "KNR Constructions Ltd",            "2008-05-02", "Infra"),
    StockInfo("NSE:PNC-EQ",         "INE474L01011", "PNC",         "PNC Infratech Ltd",                "2015-05-26", "Infra"),
    StockInfo("NSE:PRAJIND-EQ",     "INE074B01013", "PRAJIND",     "Praj Industries Ltd",              "1995-03-01", "Infra"),
    StockInfo("NSE:IRB-EQ",         "INE821I01022", "IRB",         "IRB Infrastructure Developers",    "2008-02-25", "Infra"),
    StockInfo("NSE:ASHIANA-EQ",     "INE365D01021", "ASHIANA",     "Ashiana Housing Ltd",              "1995-03-01", "Infra"),
    StockInfo("NSE:SUNTECK-EQ",     "INE805D01026", "SUNTECK",     "Sunteck Realty Ltd",               "1995-03-01", "Infra"),
    StockInfo("NSE:OBEROIRLTY-EQ",  "INE093I01010", "OBEROIRLTY",  "Oberoi Realty Ltd",                "2010-10-20", "Infra"),
    StockInfo("NSE:GODREJPROP-EQ",  "INE484J01027", "GODREJPROP",  "Godrej Properties Ltd",            "2010-01-05", "Infra"),
    StockInfo("NSE:DLF-EQ",         "INE271C01023", "DLF",         "DLF Ltd",                          "2007-07-05", "Infra"),
    StockInfo("NSE:PRESTIGE-EQ",    "INE811K01011", "PRESTIGE",    "Prestige Estates Projects Ltd",    "2010-10-27", "Infra"),
    StockInfo("NSE:PHOENIXLTD-EQ",  "INE484H01027", "PHOENIXLTD",  "Phoenix Mills Ltd",                "1995-03-01", "Infra"),
    StockInfo("NSE:BRIGADE-EQ",     "INE791I01019", "BRIGADE",     "Brigade Enterprises Ltd",          "2007-12-31", "Infra"),
    StockInfo("NSE:SOBHA-EQ",       "INE671H01015", "SOBHA",       "Sobha Ltd",                        "2006-12-26", "Infra"),
    StockInfo("NSE:MAHLIFE-EQ",     "INE766P01016", "MAHLIFE",     "Mahindra Lifespace Developers",    "2000-05-30", "Infra"),
    StockInfo("NSE:NBCC-EQ",        "INE095N01031", "NBCC",        "NBCC (India) Ltd",                 "2012-04-12", "Infra"),
    StockInfo("NSE:RITES-EQ",       "INE320J01015", "RITES",       "RITES Ltd",                        "2018-07-02", "Infra", "Listed Jul 2018"),
    StockInfo("NSE:IRCON-EQ",       "INE565M01014", "IRCON",       "Ircon International Ltd",          "2018-09-28", "Infra", "Listed Sep 2018"),
]

# =============================================================================
# SECTOR 15 — TELECOM
# =============================================================================
TELECOM = [
    StockInfo("NSE:BHARTIARTL-EQ",  "INE397D01024", "BHARTIARTL",  "Bharti Airtel Ltd",                "2002-02-18", "Telecom"),
    StockInfo("NSE:INDUSTOWER-EQ",  "INE121J01017", "INDUSTOWER",  "Indus Towers Ltd",                 "2012-12-28", "Telecom"),
    StockInfo("NSE:TATACOMM-EQ",    "INE151A01013", "TATACOMM",    "Tata Communications Ltd",          "1995-03-01", "Telecom"),
    StockInfo("NSE:STLTECH-EQ",     "INE089C01029", "STLTECH",     "STL - Sterlite Technologies",      "2000-09-08", "Telecom"),
    StockInfo("NSE:HFCL-EQ",        "INE548A01028", "HFCL",        "HFCL Ltd",                         "1995-03-01", "Telecom"),
    StockInfo("NSE:TEJASNET-EQ",    "INE578I01022", "TEJASNET",    "Tejas Networks Ltd",               "2017-06-27", "Telecom", "Listed Jun 2017"),
]

# =============================================================================
# SECTOR 16 — DEFENCE & AEROSPACE
# =============================================================================
DEFENCE = [
    StockInfo("NSE:BEL-EQ",         "INE263A01024", "BEL",         "Bharat Electronics Ltd",           "2003-08-22", "Defence"),
    StockInfo("NSE:HAL-EQ",         "INE066F01020", "HAL",         "Hindustan Aeronautics Ltd",        "2018-03-28", "Defence", "Listed Mar 2018"),
    StockInfo("NSE:BDL-EQ",         "INE171Z01018", "BDL",         "Bharat Dynamics Ltd",              "2018-03-23", "Defence", "Listed Mar 2018"),
    StockInfo("NSE:COCHINSHIP-EQ",  "INE704P01017", "COCHINSHIP",  "Cochin Shipyard Ltd",              "2017-08-11", "Defence", "Listed Aug 2017"),
    StockInfo("NSE:MAZDOCK-EQ",     "INE249Z01012", "MAZDOCK",     "Mazagon Dock Shipbuilders",        "2020-10-12", "Defence", "Listed Oct 2020"),
    StockInfo("NSE:GRSE-EQ",        "INE684A01014", "GRSE",        "Garden Reach Shipbuilders",        "2018-10-10", "Defence", "Listed Oct 2018"),
    StockInfo("NSE:BEML-EQ",        "INE258A01016", "BEML",        "BEML Ltd",                         "1995-03-01", "Defence"),
    StockInfo("NSE:MIDHANI-EQ",   "INE099Z01011", "MIDHANI",   "Mishra Dhatu Nigam Ltd (MIDHANI)", "2018-04-04", "Defence", "Listed Apr 2018"),
    StockInfo("NSE:PARAS-EQ",       "INE376P01026", "PARAS",       "Paras Defence & Space Tech",       "2021-10-01", "Defence", "Listed Oct 2021"),
    StockInfo("NSE:DATAPATTNS-EQ",  "INE863N01012", "DATAPATTNS",  "Data Patterns (India) Ltd",        "2021-12-24", "Defence", "Listed Dec 2021"),
]

# =============================================================================
# SECTOR 17 — AVIATION & LOGISTICS
# =============================================================================
AVIATION = [
    StockInfo("NSE:INDIGO-EQ",      "INE646L01027", "INDIGO",      "InterGlobe Aviation Ltd",          "2015-11-10", "Aviation"),
    StockInfo("NSE:BLUEDART-EQ",    "INE233B01017", "BLUEDART",    "Blue Dart Express Ltd",            "1995-03-01", "Aviation"),
    StockInfo("NSE:CONCOR-EQ",      "INE111A01017", "CONCOR",      "Container Corporation of India",   "1995-03-01", "Aviation"),
    StockInfo("NSE:MAHLOG-EQ",      "INE258B01022", "MAHLOG",      "Mahindra Logistics Ltd",           "2017-11-10", "Aviation", "Listed Nov 2017"),
    StockInfo("NSE:GATI-EQ",        "INE194B01030", "GATI",        "Gati Ltd",                         "1995-03-01", "Aviation"),
    StockInfo("NSE:TCI-EQ",         "INE688A01022", "TCI",         "Transport Corporation of India",   "1995-03-01", "Aviation"),
    StockInfo("NSE:ALLCARGO-EQ",    "INE418H01029", "ALLCARGO",    "Allcargo Logistics Ltd",           "2004-05-25", "Aviation"),
]

# =============================================================================
# SECTOR 18 — ADANI GROUP (pre-2017 listings only, flagged others)
# =============================================================================
ADANI = [
    StockInfo("NSE:ADANIENT-EQ",    "INE423A01024", "ADANIENT",    "Adani Enterprises Ltd",            "1994-09-15", "Adani Group"),
    StockInfo("NSE:ADANIPORTS-EQ",  "INE742F01042", "ADANIPORTS",  "Adani Ports & SEZ Ltd",            "2007-11-27", "Adani Group"),
    StockInfo("NSE:ADANIPOWER-EQ",  "INE814H01011", "ADANIPOWER",  "Adani Power Ltd",                  "2009-08-20", "Adani Group"),  # dup from POWER — deduplicated
    StockInfo("NSE:ADANIENSOL-EQ",  "INE931S01010", "ADANIENSOL",  "Adani Energy Solutions Ltd",           "2015-08-06", "Adani Group"),
    StockInfo("NSE:ADANIGREEN-EQ",  "INE364U01010", "ADANIGREEN",  "Adani Green Energy Ltd",           "2018-06-18", "Adani Group"),  # dup from POWER
]

# =============================================================================
# SECTOR 19 — CHEMICALS & SPECIALTY CHEMICALS
# =============================================================================
CHEMICALS = [
    StockInfo("NSE:PIDILITIND-EQ",  "INE318A01026", "PIDILITIND",  "Pidilite Industries Ltd",          "1998-07-21", "Chemicals"),
    StockInfo("NSE:SRF-EQ",         "INE647A01010", "SRF",         "SRF Ltd",                          "1995-03-01", "Chemicals"),
    # AAVAS moved to HOUSING_FINANCE sector (was incorrectly placed here)
    StockInfo("NSE:ATUL-EQ",        "INE100A01010", "ATUL",        "Atul Ltd",                         "1995-03-01", "Chemicals"),
    StockInfo("NSE:NAVINFLUOR-EQ",  "INE048G01026", "NAVINFLUOR",  "Navin Fluorine International",     "1994-09-15", "Chemicals"),
    StockInfo("NSE:DEEPAKNTR-EQ",   "INE288B01029", "DEEPAKNTR",   "Deepak Nitrite Ltd",               "2008-11-12", "Chemicals"),
    StockInfo("NSE:FINEORG-EQ",     "INE742C01015", "FINEORG",     "Fine Organic Industries Ltd",      "2018-06-25", "Chemicals", "Listed Jun 2018"),
    StockInfo("NSE:GALAXYSURF-EQ",  "INE418E01017", "GALAXYSURF",  "Galaxy Surfactants Ltd",           "2018-02-08", "Chemicals", "Listed Feb 2018"),
    StockInfo("NSE:CLEAN-EQ",       "INE825L01020", "CLEAN",       "Clean Science & Technology",       "2021-07-19", "Chemicals", "Listed Jul 2021"),
    StockInfo("NSE:TATACHEM-EQ",    "INE092A01019", "TATACHEM",    "Tata Chemicals Ltd",               "1996-07-01", "Chemicals"),
    StockInfo("NSE:GNFC-EQ",        "INE113A01013", "GNFC",        "Gujarat Narmada Valley Fert Co",   "1995-03-01", "Chemicals"),
    StockInfo("NSE:COROMANDEL-EQ",  "INE169A01031", "COROMANDEL",  "Coromandel International Ltd",     "1998-06-12", "Chemicals"),
    StockInfo("NSE:CHAMBLFERT-EQ",  "INE085A01013", "CHAMBLFERT",  "Chambal Fertilisers & Chem",       "1995-03-01", "Chemicals"),
    StockInfo("NSE:GSFC-EQ",        "INE026A01025", "GSFC",        "Gujarat State Fertilizers & Chem", "1995-03-01", "Chemicals"),
    StockInfo("NSE:VINATIORGA-EQ",  "INE048G01018", "VINATIORGA",  "Vinati Organics Ltd",              "2006-01-09", "Chemicals"),
    StockInfo("NSE:SUMICHEM-EQ",    "INE251B01027", "SUMICHEM",    "Sumitomo Chemical India Ltd",      "2021-09-14", "Chemicals", "Listed Sep 2021"),
    StockInfo("NSE:FLUOROCHEM-EQ",  "INE854B01025", "FLUOROCHEM",  "Gujarat Fluorochemicals Ltd",      "2021-10-14", "Chemicals", "Listed Oct 2021"),
    StockInfo("NSE:BAYERCROP-EQ",   "INE462A01022", "BAYERCROP",   "Bayer CropScience Ltd",            "1998-11-23", "Chemicals"),
    StockInfo("NSE:PIIND-EQ",       "INE603J01030", "PIIND",       "PI Industries Ltd",                "1995-03-01", "Chemicals"),
    StockInfo("NSE:UPL-EQ",         "INE628A01036", "UPL",         "UPL Ltd",                          "1995-03-01", "Chemicals"),
    StockInfo("NSE:DHANUKA-EQ",     "INE435G01025", "DHANUKA",     "Dhanuka Agritech Ltd",             "2010-08-03", "Chemicals"),
    StockInfo("NSE:LXCHEM-EQ",      "INE00SB01010", "LXCHEM",      "Laxmi Organic Industries Ltd",    "2021-03-25", "Chemicals", "Listed Mar 2021"),
    StockInfo("NSE:ASAHIINDIA-EQ",  "INE439A01020", "ASAHIINDIA",  "Asahi India Glass Ltd",            "1999-06-15", "Chemicals"),
    StockInfo("NSE:NOCIL-EQ",       "INE163A01018", "NOCIL",       "NOCIL Ltd",                        "1995-03-01", "Chemicals"),
]

# =============================================================================
# SECTOR 20 — TEXTILES & APPAREL
# =============================================================================
TEXTILES = [
    StockInfo("NSE:PAGEIND-EQ",     "INE761H01022", "PAGEIND",     "Page Industries Ltd",              "2007-03-16", "Textiles"),  # dup
    StockInfo("NSE:RAYMOND-EQ",     "INE301A01014", "RAYMOND",     "Raymond Ltd",                      "1995-03-01", "Textiles"),
    StockInfo("NSE:TRIDENT-EQ",     "INE064C01022", "TRIDENT",     "Trident Ltd",                      "1997-04-01", "Textiles"),
    StockInfo("NSE:VTL-EQ","INE825A01012", "VTL","Vardhman Textiles Ltd",             "1995-03-01", "Textiles"),
    StockInfo("NSE:WELSPUNIND-EQ",  "INE192B01031", "WELSPUNIND",  "Welspun India Ltd",                "2004-12-23", "Textiles"),
    StockInfo("NSE:ARVIND-EQ",      "INE034A01011", "ARVIND",      "Arvind Ltd",                       "1995-03-01", "Textiles"),
    StockInfo("NSE:KPRMILL-EQ",         "INE930H01023", "KPRMILL",         "KPR Mill Ltd",                     "2008-10-13", "Textiles"),
    StockInfo("NSE:GARFIBRES-EQ",   "INE409B01013", "GARFIBRES",   "Garware Technical Fibres Ltd",     "1998-01-05", "Textiles"),
    # SPANDANA moved to NBFC2 sector (was incorrectly placed in Textiles)
]

# =============================================================================
# SECTOR 21 — ENGINEERING, CAPITAL GOODS & INDUSTRIALS
# =============================================================================
ENGINEERING = [
    StockInfo("NSE:SIEMENS-EQ",     "INE003A01024", "SIEMENS",     "Siemens Ltd",                      "1998-04-14", "Engineering", "Energy biz demerged Jun 2025 (SEIL); parent SIEMENS continues"),
    StockInfo("NSE:ABB-EQ",         "INE117A01022", "ABB",         "ABB India Ltd",                    "1995-03-01", "Engineering"),
    StockInfo("NSE:HAVELLS-EQ",     "INE176B01034", "HAVELLS",     "Havells India Ltd",                "1994-09-15", "Engineering"),
    StockInfo("NSE:HONAUT-EQ",      "INE671A01010", "HONAUT",      "Honeywell Automation India Ltd",   "1995-03-01", "Engineering"),
    StockInfo("NSE:BHEL-EQ",        "INE257A01026", "BHEL",        "Bharat Heavy Electricals Ltd",     "1994-09-15", "Engineering"),
    StockInfo("NSE:CUMMINSIND-EQ",  "INE298A01020", "CUMMINSIND",  "Cummins India Ltd",                "1995-03-01", "Engineering"),
    StockInfo("NSE:THERMAX-EQ",     "INE152A01029", "THERMAX",     "Thermax Ltd",                      "2007-12-31", "Engineering"),
    StockInfo("NSE:AIAENG-EQ",      "INE212H01026", "AIAENG",      "AIA Engineering Ltd",              "2006-01-09", "Engineering"),
    StockInfo("NSE:GRINDWELL-EQ",   "INE536A01023", "GRINDWELL",   "Grindwell Norton Ltd",             "1995-03-01", "Engineering"),
    StockInfo("NSE:CARBORUNIV-EQ",  "INE813A01018", "CARBORUNIV",  "Carborundum Universal Ltd",        "1995-03-01", "Engineering"),
    StockInfo("NSE:SKFINDIA-EQ",    "INE640A01023", "SKFINDIA",    "SKF India Ltd",                    "1995-03-01", "Engineering"),
    StockInfo("NSE:TIMKEN-EQ",      "INE325A01013", "TIMKEN",      "Timken India Ltd",                 "2002-02-19", "Engineering"),
    StockInfo("NSE:ELGIEQUIP-EQ",   "INE285A01027", "ELGIEQUIP",   "Elgi Equipments Ltd",              "1995-03-01", "Engineering"),
    StockInfo("NSE:KSB-EQ",         "INE999A01015", "KSB",         "KSB Ltd",                          "1995-03-01", "Engineering"),
    StockInfo("NSE:VOLTAMP-EQ",     "INE685E01016", "VOLTAMP",     "Voltamp Transformers Ltd",         "2002-09-04", "Engineering"),
    StockInfo("NSE:CGPOWER-EQ",     "INE067A01029", "CGPOWER",     "CG Power & Industrial Solutions",  "1995-03-01", "Engineering"),
    StockInfo("NSE:INOXINDIA-EQ",   "INE549W01016", "INOXINDIA",   "INOX India Ltd",                   "2023-12-21", "Engineering", "Listed Dec 2023; post-COVID only"),
    StockInfo("NSE:KAYNES-EQ",      "INE918Z01012", "KAYNES",      "Kaynes Technology India Ltd",      "2022-11-22", "Engineering", "Listed Nov 2022"),
    StockInfo("NSE:VIPIND-EQ",        "INE560A01015", "VIPIND",        "VIP Industries Ltd",                   "1995-03-01", "Retail",           "Pre-2017; full history; luggage mfg"),
    StockInfo("NSE:ESCORTS-EQ",     "INE042A01014", "ESCORTS",     "Escorts Kubota Ltd",               "1995-03-01", "Engineering"),
    StockInfo("NSE:TEXRAIL-EQ",     "INE523H01014", "TEXRAIL",     "Texmaco Rail & Engineering Ltd",   "2010-05-31", "Engineering"),
    StockInfo("NSE:RVNL-EQ",        "INE415G01027", "RVNL",        "Rail Vikas Nigam Ltd",             "2019-04-11", "Engineering", "Listed Apr 2019; minimal pre-COVID"),
    StockInfo("NSE:IRFC-EQ",        "INE053F01010", "IRFC",        "Indian Railway Finance Corp",      "2021-01-29", "Engineering", "Listed Jan 2021; post-COVID + partial pre"),
    StockInfo("NSE:TITAGARH-EQ",    "INE953D01020", "TITAGARH",    "Titagarh Rail Systems Ltd",        "2000-02-14", "Engineering"),
    StockInfo("NSE:ISGEC-EQ",       "INE411C01022", "ISGEC",       "ISGEC Heavy Engineering Ltd",      "1995-03-01", "Engineering"),
]

# =============================================================================
# SECTOR 22 — MEDIA & ENTERTAINMENT
# =============================================================================
MEDIA = [
    StockInfo("NSE:ZEEL-EQ",        "INE256A01028", "ZEEL",        "Zee Entertainment Enterprises",    "1995-03-01", "Media"),
    StockInfo("NSE:SUNTV-EQ",       "INE649A01019", "SUNTV",       "Sun TV Network Ltd",               "2006-04-24", "Media"),
    StockInfo("NSE:PVRINOX-EQ",     "INE191H01014", "PVRINOX",     "PVR Inox Ltd",                     "2006-01-31", "Media"),
    StockInfo("NSE:WONDERLA-EQ",      "INE066I01010", "WONDERLA",      "Wonderla Holidays Ltd",                "2014-05-09", "Retail",          "Amusement parks; full pre-COVID history"),
    StockInfo("NSE:TIPSFILMS-EQ",        "INE716B01011", "TIPSFILMS",        "Tips Music Ltd",                   "2000-09-11", "Media"),
    StockInfo("NSE:SAREGAMA-EQ",    "INE979A01025", "SAREGAMA",    "Saregama India Ltd",               "1995-03-01", "Media"),
    StockInfo("NSE:JAGRAN-EQ",      "INE199G01027", "JAGRAN",      "Jagran Prakashan Ltd",             "2010-07-01", "Media"),
    StockInfo("NSE:DBCORP-EQ",      "INE950I01011", "DBCORP",      "DB Corp Ltd",                      "2010-01-06", "Media"),
]

# =============================================================================
# SECTOR 23 — CONSUMER DURABLES & HOME APPLIANCES
# =============================================================================
DURABLES = [
    StockInfo("NSE:VOLTAS-EQ",      "INE226A01021", "VOLTAS",      "Voltas Ltd",                       "1995-03-01", "Durables"),
    StockInfo("NSE:BLUESTARCO-EQ",    "INE472A01039", "BLUESTARCO",    "Blue Star Ltd",                    "1995-03-01", "Durables"),
    StockInfo("NSE:CROMPTON-EQ",    "INE299U01018", "CROMPTON",    "Crompton Greaves Consumer Elec",   "2016-05-13", "Durables"),
    StockInfo("NSE:BAJAJELE-EQ", "INE193E01017", "BAJAJELE", "Bajaj Electricals Ltd",            "1995-03-01", "Durables"),
    StockInfo("NSE:RAJESHEXPO-EQ",  "INE251D01021", "RAJESHEXPO",  "Rajesh Exports Ltd",               "1995-03-01", "Durables"),
    StockInfo("NSE:ORIENTELEC-EQ",  "INE142B01024", "ORIENTELEC",  "Orient Electric Ltd",              "2018-05-03", "Durables", "Listed May 2018"),
    StockInfo("NSE:AMBER-EQ",       "INE956K01015", "AMBER",       "Amber Enterprises India Ltd",      "2018-01-30", "Durables", "Listed Jan 2018"),
    StockInfo("NSE:DIXON-EQ",       "INE935N01020", "DIXON",       "Dixon Technologies Ltd",           "2017-09-18", "Durables", "Listed Sep 2017"),
    StockInfo("NSE:WHIRLPOOL-EQ",   "INE716A01013", "WHIRLPOOL",   "Whirlpool of India Ltd",           "1995-03-01", "Durables"),
    StockInfo("NSE:TITAN-EQ",       "INE280A01028", "TITAN",       "Titan Company Ltd",                "1995-03-01", "Durables"),  # dup — removed
    StockInfo("NSE:KALYANKJIL-EQ",      "INE303R01014", "KALYANKJIL",      "Kalyan Jewellers India Ltd",       "2021-03-26", "Durables", "Listed Mar 2021"),
    StockInfo("NSE:SENCO-EQ",       "INE03HV01017", "SENCO",       "Senco Gold Ltd",                   "2023-07-14", "Durables", "Listed Jul 2023"),
]

# =============================================================================
# SECTOR 24 — HOSPITALS & DIAGNOSTICS (additional)
# =============================================================================
HOSPITALS = [
    StockInfo("NSE:KIMS-EQ",        "INE836S01028", "KIMS",        "Krishna Institute of Med Sciences","2021-06-28", "Healthcare", "Listed Jun 2021"),
    StockInfo("NSE:RAINBOW-EQ",     "INE274L01014", "RAINBOW",     "Rainbow Children Medicare Ltd",    "2022-05-10", "Healthcare", "Listed May 2022"),
    StockInfo("NSE:ERIS-EQ",        "INE406M01024", "ERIS",        "Eris Lifesciences Ltd",            "2017-06-29", "Healthcare", "Listed Jun 2017"),
    StockInfo("NSE:THYROCARE-EQ",   "INE069I01010", "THYROCARE",   "Thyrocare Technologies Ltd",       "2016-05-09", "Healthcare"),
    StockInfo("NSE:KRSNAA-EQ",      "INE136V01027", "KRSNAA",      "Krsnaa Diagnostics Ltd",           "2021-08-16", "Healthcare", "Listed Aug 2021"),
    StockInfo("NSE:VIJAYA-EQ",      "INE073D01013", "VIJAYA",      "Vijaya Diagnostic Centre Ltd",     "2021-09-14", "Healthcare", "Listed Sep 2021"),
]

# =============================================================================
# SECTOR 25 — MISCELLANEOUS / DIVERSIFIED
# =============================================================================
MISC = [
    StockInfo("NSE:IEX-EQ",         "INE022Q01020", "IEX",         "Indian Energy Exchange Ltd",       "2017-10-23", "Misc", "Listed Oct 2017"),
    StockInfo("NSE:CDSL-EQ",        "INE736A01011", "CDSL",        "Central Depository Services",      "2017-06-30", "Misc"),  # dup with Capital Markets
    StockInfo("NSE:IRCTC-EQ",       "INE335Y01020", "IRCTC",       "Indian Railway Catering & Tourism","2019-10-14", "Misc"),
    StockInfo("NSE:NYKAA-EQ",       "INE388Y01029", "NYKAA",       "FSN E-Commerce Ventures (Nykaa)",  "2021-11-10", "Misc", "Listed Nov 2021; post-2022 data only"),
    StockInfo("NSE:POLICYBZR-EQ",   "INE417T01026", "POLICYBZR",   "PB Fintech (PolicyBazaar)",        "2021-11-15", "Misc", "Listed Nov 2021"),
    StockInfo("NSE:CARTRADE-EQ",    "INE018L01020", "CARTRADE",    "CarTrade Tech Ltd",                "2021-08-20", "Misc", "Listed Aug 2021"),
    StockInfo("NSE:JUSTDIAL-EQ",    "INE599M01018", "JUSTDIAL",    "Just Dial Ltd",                    "2013-06-05", "Misc"),
    StockInfo("NSE:AFFLE-EQ",       "INE00WJ01018", "AFFLE",       "Affle (India) Ltd",                "2019-08-08", "Misc"),
    StockInfo("NSE:INDIGOPNTS-EQ",  "INE09SL01010", "INDIGOPNTS",  "Indigo Paints Ltd",                "2021-02-02", "Misc", "Listed Feb 2021"),
    StockInfo("NSE:EPIGRAL-EQ",     "INE045W01011", "EPIGRAL",     "Epigral Ltd",                      "2010-04-01", "Misc"),
    StockInfo("NSE:SIGNATURE-EQ",   "INE060F01028", "SIGNATURE",   "Signature Global India Ltd",       "2023-09-27", "Misc", "Listed Sep 2023"),
    StockInfo("NSE:GMRAIRPORT-EQ",    "INE776C01039", "GMRAIRPORT",    "GMR Airports Infrastructure Ltd",      "2006-08-22", "Misc"),
    StockInfo("NSE:AMIORG-EQ",      "INE00BI01011", "AMIORG",      "Ami Organics Ltd",                 "2021-09-14", "Misc", "Listed Sep 2021"),
    StockInfo("NSE:TATAINVEST-EQ",  "INE672A01018", "TATAINVEST",  "Tata Investment Corporation",      "1995-03-01", "Misc"),
    StockInfo("NSE:KFINTECH-EQ",    "INE138Y01010", "KFINTECH",    "KFin Technologies Ltd",            "2021-12-29", "Misc", "Listed Dec 2021"),
    StockInfo("NSE:GPIL-EQ",        "INE177H01026", "GPIL",        "Godawari Power & Ispat Ltd",       "2010-04-06", "Misc"),
    StockInfo("NSE:NESCO-EQ",       "INE317A01021", "NESCO",       "NESCO Ltd",                        "1995-03-01", "Misc"),
    StockInfo("NSE:NAZARA-EQ",      "INE418L01020", "NAZARA",      "Nazara Technologies Ltd",          "2021-03-30", "Misc", "Listed Mar 2021"),
    StockInfo("NSE:LATENTVIEW-EQ",  "INE0MK701018", "LATENTVIEW",  "Latent View Analytics Ltd",        "2021-11-23", "Misc", "Listed Nov 2021"),
    StockInfo("NSE:MTAR-EQ",        "INE587K01014", "MTAR",        "MTAR Technologies Ltd",            "2021-03-15", "Misc", "Listed Mar 2021"),
]


# =============================================================================
# SECTOR 26 — HOUSING FINANCE
# =============================================================================
HOUSING_FINANCE = [
    StockInfo("NSE:AAVAS-EQ",       "INE216P01012", "AAVAS",       "Aavas Financiers Ltd",             "2018-10-08", "Housing Finance", "Listed Oct 2018"),
    StockInfo("NSE:HOMEFIRST-EQ",   "INE481M01028", "HOMEFIRST",   "Home First Finance Company India", "2021-02-03", "Housing Finance", "Listed Feb 2021"),
    StockInfo("NSE:APTUS-EQ",       "INE852O01025", "APTUS",       "Aptus Value Housing Finance",      "2021-08-24", "Housing Finance", "Listed Aug 2021"),
    StockInfo("NSE:REPCOHOME-EQ",       "INE612J01015", "REPCOHOME",       "Repco Home Finance Ltd",           "2013-04-01", "Housing Finance"),
    StockInfo("NSE:AADHARHFC-EQ",     "INE00J801013", "AADHARHFC",     "Aadhar Housing Finance Ltd",           "2024-05-23", "Housing Finance",  "Listed May 2024; post-COVID only"),
]

# =============================================================================
# SECTOR 27 — MICROFINANCE
# =============================================================================
MICROFINANCE = [
    StockInfo("NSE:MFSL-EQ",        "INE103G01010", "MFSL",        "Max Financial Services Ltd",       "2000-07-20", "Insurance"),
    StockInfo("NSE:FUSION-EQ",      "INE139R01013", "FUSION",      "Fusion Micro Finance Ltd",         "2022-11-02", "Microfinance", "Listed Nov 2022"),
]

# =============================================================================
# SECTOR 28 — PAINTS & COATINGS
# =============================================================================
PAINTS = [
    StockInfo("NSE:BERGEPAINT-EQ",      "INE463A01038", "BERGEPAINT",      "Berger Paints India Ltd",          "1995-03-01", "Paints"),
    StockInfo("NSE:KANSAINER-EQ",   "INE613A01020", "KANSAINER",   "Kansai Nerolac Paints Ltd",        "1995-03-01", "Paints"),
    StockInfo("NSE:AKZOINDIA-EQ",   "INE133A01011", "AKZOINDIA",   "Akzo Nobel India Ltd",             "1995-03-01", "Paints"),
    # INDIGO removed from Paints list — InterGlobe Aviation already in AVIATION sector correctly
]

# =============================================================================
# SECTOR 29 — SUGAR & AGRO
# =============================================================================
SUGAR_AGRO = [
    StockInfo("NSE:BALRAMCHIN-EQ",  "INE119A01028", "BALRAMCHIN",  "Balrampur Chini Mills Ltd",        "1995-03-01", "Sugar"),
    StockInfo("NSE:RENUKA-EQ",      "INE087H01022", "RENUKA",      "Shree Renuka Sugars Ltd",          "2005-01-07", "Sugar"),
    StockInfo("NSE:TRIVENI-EQ",     "INE256C01024", "TRIVENI",     "Triveni Engineering & Inds",       "2004-03-16", "Sugar"),
    StockInfo("NSE:DHAMPURSUG-EQ",     "INE041A01016", "DHAMPURSUG",     "Dhampur Sugar Mills Ltd",          "1995-03-01", "Sugar"),
    StockInfo("NSE:EIDPARRY-EQ",    "INE126A01031", "EIDPARRY",    "EID Parry (India) Ltd",            "1995-03-01", "Sugar"),
    StockInfo("NSE:AVANTIFEED-EQ",  "INE871C01038", "AVANTIFEED",  "Avanti Feeds Ltd",                 "1995-03-01", "Agro"),
    StockInfo("NSE:KRBL-EQ",        "INE001B01026", "KRBL",        "KRBL Ltd",                         "1995-03-01", "Agro"),
    StockInfo("NSE:LT-EQ",          "INE018A01030", "LT",          "Larsen & Toubro Ltd",              "1995-03-01", "Infra"),  # dup
    StockInfo("NSE:DEVYANI-EQ",       "INE977E01020", "DEVYANI",       "Devyani International Ltd",            "2021-08-11", "Retail",           "QSR major (KFC/Pizza Hut); listed 2021"),
]

# =============================================================================
# SECTOR 30 — PAPER & PACKAGING
# =============================================================================
PAPER = [
    StockInfo("NSE:TNPL-EQ",        "INE890A01016", "TNPL",        "Tamil Nadu Newsprint & Papers",    "1995-03-01", "Paper"),
    StockInfo("NSE:JKPAPER-EQ",     "INE477B01010", "JKPAPER",     "JK Paper Ltd",                     "1995-03-01", "Paper"),
    StockInfo("NSE:WSTCSTPAPR-EQ",  "INE976A01021", "WSTCSTPAPR",  "West Coast Paper Mills Ltd",       "1995-03-01", "Paper"),
    StockInfo("NSE:HUHTAMAKI-EQ",   "INE056B01011", "HUHTAMAKI",   "Huhtamaki India Ltd",              "1995-03-01", "Paper"),
    StockInfo("NSE:MOLDTKPAC-EQ",    "INE413C01014", "MOLDTKPAC",    "Mold-Tek Packaging Ltd",           "1995-03-01", "Paper"),
    StockInfo("NSE:UFLEX-EQ",       "INE516A01017", "UFLEX",       "UFLEX Ltd",                        "1995-03-01", "Paper"),
    StockInfo("NSE:TDPOWERSYS-EQ",  "INE941P01014", "TDPOWERSYS",  "TD Power Systems Ltd",             "2011-09-14", "Engineering"),
]

# =============================================================================
# SECTOR 31 — GEMS, JEWELLERY & WATCHES (additional)
# =============================================================================
GEMS = [
    StockInfo("NSE:MMTC-EQ",          "INE123F01029", "MMTC",          "MMTC Ltd",                             "2004-06-26", "Misc",             "Pre-2017; govt commodities trading PSU"),
    StockInfo("NSE:PCJEWELLER-EQ",  "INE785L01026", "PCJEWELLER",  "PC Jeweller Ltd",                  "2012-12-10", "Gems"),
    StockInfo("NSE:THANGAMAYL-EQ",  "INE071G01030", "THANGAMAYL",  "Thangamayil Jewellery Ltd",        "2010-04-01", "Gems"),
    StockInfo("NSE:RAJESHEXPO-EQ",  "INE251D01021", "RAJESHEXPO",  "Rajesh Exports Ltd",               "1995-03-01", "Gems"),  # dup
]

# =============================================================================
# SECTOR 32 — IT (ADDITIONAL / MIDCAP)
# =============================================================================
IT_MID = [
    StockInfo("NSE:ZENSARTECH-EQ",      "INE520A01027", "ZENSARTECH",      "Zensar Technologies Ltd",          "2004-08-27", "IT"),
    StockInfo("NSE:NIIT-EQ",        "INE236A01020", "NIIT",        "NIIT Ltd",                         "1993-07-29", "IT"),
    StockInfo("NSE:BIRLASOFT-EQ",   "INE005I01014", "BIRLASOFT",   "Birlasoft Ltd",                    "2004-08-27", "IT"),
    StockInfo("NSE:GALAXYSURF-EQ",  "INE418E01017", "GALAXYSURF",  "Galaxy Surfactants Ltd",           "2018-02-08", "Chemicals"),  # dup
    StockInfo("NSE:BSOFT-EQ",         "INE096F01010", "BSOFT",         "Birlasoft Ltd",                        "2004-08-27", "IT",               "Pre-2017; full history; IT services"),
    StockInfo("NSE:SUBEXLTD-EQ",    "INE754A01014", "SUBEXLTD",    "Subex Ltd",                        "2000-02-17", "IT"),
    StockInfo("NSE:NUCLEUS-EQ",     "INE096B01018", "NUCLEUS",     "Nucleus Software Exports Ltd",     "1996-01-15", "IT"),
    StockInfo("NSE:INFOBEAN-EQ",    "INE483B01020", "INFOBEAN",    "InfoBeans Technologies Ltd",       "2009-03-27", "IT"),
    StockInfo("NSE:SAKSOFT-EQ",     "INE667G01014", "SAKSOFT",     "Saksoft Ltd",                      "2000-11-07", "IT"),
    StockInfo("NSE:WIPRO-EQ",       "INE075A01022", "WIPRO",       "Wipro Ltd",                        "1995-01-13", "IT"),  # dup
    StockInfo("NSE:ECLERX-EQ",      "INE738I01010", "ECLERX",      "eClerx Services Ltd",              "2007-12-31", "IT"),
    StockInfo("NSE:TATATECH-EQ",    "INE142M01025", "TATATECH",    "Tata Technologies Ltd",            "2023-11-30", "IT", "Listed Nov 2023"),
    StockInfo("NSE:QUICKHEAL-EQ",   "INE306L01010", "QUICKHEAL",   "Quick Heal Technologies Ltd",      "2016-02-18", "IT"),
    StockInfo("NSE:SAKUMA-EQ",      "INE667I01014", "SAKUMA",      "Sakuma Exports Ltd",               "2010-04-01", "IT"),
    StockInfo("NSE:INFIBEAM-EQ",    "INE483S01020", "INFIBEAM",    "Infibeam Avenues Ltd",             "2016-04-21", "IT"),
    StockInfo("NSE:RAMKY-EQ",         "INE874I01013", "RAMKY",         "Ramky Infrastructure Ltd",             "2010-09-10", "Infra",            "Pre-2017; full history; EPC infra"),
]

# =============================================================================
# SECTOR 33 — NBFC (ADDITIONAL)
# =============================================================================
NBFC2 = [
    # SRTRANSFIN removed — old ticker for SHRIRAMFIN (same company); duplicate ISIN INE721A01013
    StockInfo("NSE:BAJAJHFL-EQ",    "INE913H01037", "BAJAJHFL",    "Bajaj Housing Finance Ltd",        "2024-09-16", "NBFC", "Listed Sep 2024"),
    StockInfo("NSE:EDELWEISS-EQ",     "INE532F01054", "EDELWEISS",     "Edelweiss Financial Services Ltd",     "2007-11-28", "Capital Markets",  "Pre-2017; full history"),
    StockInfo("NSE:IBULHSGFIN-EQ",  "INE148I01020", "IBULHSGFIN",  "Indiabulls Housing Finance",       "2005-07-07", "NBFC"),
    StockInfo("NSE:INDIAGLYCO-EQ",  "INE204A01010", "INDIAGLYCO",  "India Glycols Ltd",                "2000-11-21", "Chemicals"),
    StockInfo("NSE:APTECHT-EQ",     "INE266F01018", "APTECHT",     "Aptech Ltd",                       "1994-09-15", "IT"),
    StockInfo("NSE:BIRLACORPN-EQ",   "INE340A01012", "BIRLACORPN",    "Birla Corporation Ltd",                "1997-04-07", "Cement",           "Pre-2017; full history"),
    StockInfo("NSE:SPANDANA-EQ",    "INE572J01011", "SPANDANA",    "Spandana Sphoorty Financial Ltd",  "2019-08-19", "NBFC", "Microfinance NBFC"),
    StockInfo("NSE:FIVESTAR-EQ",   "INE128S01021", "FIVESTAR",   "Five-Star Business Finance Ltd",   "2022-11-21", "NBFC", "Listed Nov 2022"),
    StockInfo("NSE:UGROCAP-EQ",     "INE551W01018", "UGROCAP",     "Ugro Capital Ltd",                 "2018-08-22", "NBFC"),  # dup
]

# =============================================================================
# SECTOR 34 — HOSPITALS / HEALTHCARE ADDITIONAL
# =============================================================================
HEALTH2 = [
    StockInfo("NSE:POLYMED-EQ",     "INE205C01021", "POLYMED",     "Poly Medicure Ltd",                "2004-03-26", "Healthcare"),
    StockInfo("NSE:ASTERDM-EQ",     "INE914M01019", "ASTERDM",     "Aster DM Healthcare Ltd",          "2018-02-26", "Healthcare", "Listed Feb 2018"),
    StockInfo("NSE:MEDANTA-EQ",     "INE0NAA01012", "MEDANTA",     "Global Health Ltd",                "2022-11-16", "Healthcare", "Listed Nov 2022"),
    # JUPITERHOS removed — was incorrectly mapped to Jupiter Wagons (Engineering). 
    # KIMS (Krishna Institute of Med Sciences) is the correct Healthcare entry at INE836S01028.
    StockInfo("NSE:HEALTHCARE-EQ",  "INE488C01018", "HEALTHCARE",  "Healthcare Global Enterprises",    "2016-03-24", "Healthcare"),
    StockInfo("NSE:SUPRIYA-EQ",     "INE02XI01013", "SUPRIYA",     "Supriya Lifescience Ltd",          "2021-12-28", "Pharma", "Listed Dec 2021"),
    StockInfo("NSE:JBCHEPHARM-EQ",  "INE047B01015", "JBCHEPHARM",  "JB Chemicals & Pharmaceuticals",   "1995-03-01", "Pharma"),
    StockInfo("NSE:WOCKPHARMA-EQ",  "INE049B01025", "WOCKPHARMA",  "Wockhardt Ltd",                    "1999-11-29", "Pharma"),
    StockInfo("NSE:STAR-EQ",     "INE939A01011", "STAR",     "Strides Pharma Science Ltd",       "1999-11-01", "Pharma"),
    StockInfo("NSE:SEQUENT-EQ",     "INE807F01027", "SEQUENT",     "SeQuent Scientific Ltd",           "2000-11-07", "Pharma"),
    StockInfo("NSE:LAURUSLABS-EQ",  "INE947Q01028", "LAURUSLABS",  "Laurus Labs Ltd",                  "2016-12-19", "Pharma"),
    StockInfo("NSE:AJANTPHARM-EQ",  "INE031B01049", "AJANTPHARM",  "Ajanta Pharma Ltd",                "1995-03-01", "Pharma"),
    StockInfo("NSE:YATHARTH-EQ",    "INE0PAP01011", "YATHARTH",    "Yatharth Hospital & Trauma",       "2023-07-26", "Healthcare", "Listed Jul 2023"),
]

# =============================================================================
# SECTOR 35 — RAILWAYS & TRANSPORTATION
# =============================================================================
RAILWAYS = [
    StockInfo("NSE:RAILTEL-EQ",     "INE224H01027", "RAILTEL",     "RailTel Corporation of India",     "2021-02-26", "Railways", "Listed Feb 2021"),
    StockInfo("NSE:TEXRAIL-EQ",     "INE523H01014", "TEXRAIL",     "Texmaco Rail & Engineering",       "2010-05-31", "Railways"),  # dup
    StockInfo("NSE:IRCTC-EQ",       "INE335Y01020", "IRCTC",       "IRCTC",                            "2019-10-14", "Railways"),  # dup
    StockInfo("NSE:IRFC-EQ",        "INE053F01010", "IRFC",        "Indian Railway Finance Corp",      "2021-01-29", "Railways"),  # dup
    StockInfo("NSE:RVNL-EQ",        "INE415G01027", "RVNL",        "Rail Vikas Nigam Ltd",             "2019-04-11", "Railways"),  # dup
    StockInfo("NSE:JUPITERWAG-EQ",  "INE376P01041", "JUPITERWAG",  "Jupiter Wagons Ltd",               "2021-06-28", "Engineering"),
    StockInfo("NSE:BEML-EQ",        "INE258A01016", "BEML",        "BEML Ltd",                         "1995-03-01", "Defence"),  # dup
    StockInfo("NSE:SIYSIL-EQ",      "INE247P01010", "SIYSIL",      "Siyaram Silk Mills Ltd",           "1995-03-01", "Textiles"),
]

# =============================================================================
# SECTOR 36 — REAL ESTATE (ADDITIONAL)
# =============================================================================
REALTY2 = [
    StockInfo("NSE:MAHINDCIE-EQ",   "INE536H01010", "MAHINDCIE",   "Mahindra CIE Automotive Ltd",     "2007-05-10", "Auto"),
    StockInfo("NSE:ANANTRAJ-EQ",    "INE242C01024", "ANANTRAJ",    "Anant Raj Ltd",                    "1999-05-03", "Realty"),
    StockInfo("NSE:ELECON-EQ",        "INE225A01022", "ELECON",        "Elecon Engineering Co Ltd",            "1995-03-01", "Engineering",      "Pre-2017; full history; gears & MHE"),
    StockInfo("NSE:KOLTEPATIL-EQ",  "INE094I01010", "KOLTEPATIL",  "Kolte-Patil Developers Ltd",       "2007-12-18", "Realty"),
    StockInfo("NSE:MAHLIFE-EQ",     "INE766P01016", "MAHLIFE",     "Mahindra Lifespace Developers",    "2000-05-30", "Realty"),  # dup
    # SOBHA duplicate removed — already present in INFRA section with correct ticker
    StockInfo("NSE:NCLIND-EQ",      "INE196E01010", "NCLIND",      "NCL Industries Ltd",               "1995-03-01", "Cement"),
    StockInfo("NSE:JSWINFRA-EQ",    "INE880J01024", "JSWINFRA",    "JSW Infrastructure Ltd",           "2023-09-25", "Infra", "Listed Sep 2023"),
    StockInfo("NSE:KEYSTONEREALTY-EQ",    "INE04TQ01019", "KEYSTONEREALTY",    "Keystone Realtors Ltd",            "2022-12-14", "Realty", "Listed Dec 2022"),
    StockInfo("NSE:HEMIPROP-EQ",    "INE0GFG01019", "HEMIPROP",    "Hemisphere Properties India",      "2021-01-18", "Realty"),
    StockInfo("NSE:LODHA-EQ",   "INE670K01029", "LODHA",   "Macrotech Developers Ltd (Lodha)", "2021-04-19", "Realty", "Listed Apr 2021"),
    StockInfo("NSE:TATACONSUM-EQ",  "INE192A01025", "TATACONSUM",  "Tata Consumer Products",           "1998-11-03", "FMCG"),  # dup
]

# =============================================================================
# SECTOR 37 — DIVERSIFIED / CONGLOMERATES
# =============================================================================
CONGLOMERATES = [
    StockInfo("NSE:GODREJIND-EQ",   "INE233A01035", "GODREJIND",   "Godrej Industries Ltd",            "1995-03-01", "Conglomerate"),
    StockInfo("NSE:SONACOMS-EQ",      "INE016Q01028", "SONACOMS",      "Sona BLW Precision Forgings Ltd",      "2021-06-24", "Auto",             "Auto components; listed Jun 2021"),
    StockInfo("NSE:WABCOINDIA-EQ",   "INE342J01019", "WABCOINDIA",    "Wabco India Ltd",                      "2004-03-12", "Auto",             "Pre-2017; full history; auto components"),
    StockInfo("NSE:3MINDIA-EQ",     "INE470A01017", "3MINDIA",     "3M India Ltd",                     "2004-05-26", "Engineering"),
    StockInfo("NSE:BASF-EQ",        "INE373A01013", "BASF",        "BASF India Ltd",                   "1998-06-15", "Chemicals"),
    StockInfo("NSE:SANOFI-EQ",      "INE574A01011", "SANOFI",      "Sanofi India Ltd",                 "1995-03-01", "Pharma"),
    StockInfo("NSE:PFIZER-EQ",      "INE182A01018", "PFIZER",      "Pfizer Ltd",                       "1995-03-01", "Pharma"),
    StockInfo("NSE:GLAXO-EQ",       "INE159A01016", "GLAXO",       "GSK Pharmaceuticals Ltd",          "1995-03-01", "Pharma"),
    StockInfo("NSE:ASTRAZEN-EQ",    "INE203A01020", "ASTRAZEN",    "AstraZeneca Pharma India",         "1995-03-01", "Pharma"),
    StockInfo("NSE:TATAMETALI-EQ",  "INE486A01021", "TATAMETALI",  "Tata Metaliks Ltd",                "2007-11-02", "Metals"),
    StockInfo("NSE:GRAPHITE-EQ",    "INE452A01024", "GRAPHITE",    "Graphite India Ltd",               "1995-03-01", "Metals"),
    StockInfo("NSE:HLEGLAS-EQ",     "INE00F601020", "HLEGLAS",     "HLE Glascoat Ltd",                 "1999-11-25", "Engineering"),
    StockInfo("NSE:SUDARSCHEM-EQ",  "INE659A01023", "SUDARSCHEM",  "Sudarshan Chemical Industries",    "1998-08-31", "Chemicals"),
    StockInfo("NSE:KENNAMET-EQ",    "INE717A01029", "KENNAMET",    "Kennametal India Ltd",             "1998-12-21", "Engineering"),
    StockInfo("NSE:NAUKRI-EQ",      "INE663F01024", "NAUKRI",      "Info Edge (India) Ltd",            "2006-11-21", "Retail"),  # dup
    StockInfo("NSE:GPPL-EQ",          "INE508L01018", "GPPL",          "Gujarat Pipavav Port Ltd",             "2009-08-26", "Infra",            "Pre-2017; full history; port infra"),
    StockInfo("NSE:CAMPUS-EQ",      "INE00GY01011", "CAMPUS",      "Campus Activewear Ltd",            "2022-04-26", "Retail"),  # dup
    # DPWWORLD removed — DP World is not independently listed on NSE; was incorrectly using ADANIPORTS ISIN
]

# =============================================================================
# SECTOR 38 — POWER EQUIPMENT & RENEWABLES (ADDITIONAL)
# =============================================================================
POWER2 = [
    StockInfo("NSE:BOMDYEING-EQ",   "INE032A01023", "BOMDYEING",   "Bombay Dyeing & Mfg Co",           "1995-03-01", "Textiles"),
    StockInfo("NSE:APARINDS-EQ",      "INE372A01015", "APARINDS",      "APAR Industries Ltd",                  "2000-06-14", "Engineering",      "Pre-2017; full history; cables & conductors"),
    StockInfo("NSE:WEBSOL-EQ",        "INE467H01026", "WEBSOL",        "Websol Energy System Ltd",             "1995-03-01", "Power",            "Pre-2017; full history; solar cells mfg"),
    StockInfo("NSE:INOXGREEN-EQ",   "INE028L01027", "INOXGREEN",   "Inox Green Energy Services",       "2022-11-23", "Power", "Listed Nov 2022"),
    StockInfo("NSE:KIRLOSENG-EQ",    "INE164A01016", "KIRLOSENG",     "Kirloskar Brothers Ltd",               "2000-11-24", "Engineering",      "Pre-2017; full history; pumps & systems"),
    # PREMIER duplicate removed (already in ENGINEERING section above)
    StockInfo("NSE:POWERMECH-EQ",   "INE143K01010", "POWERMECH",   "Powermech Projects Ltd",           "2015-08-21", "Engineering"),
    StockInfo("NSE:POKARNA-EQ",       "INE143K01002", "POKARNA",       "Pokarna Ltd",                          "2001-10-31", "Building",         "Pre-2017; full history; quartz surfaces"),
    StockInfo("NSE:GENUSPOWER-EQ",  "INE454A01016", "GENUSPOWER",  "Genus Power Infrastructures",      "1998-01-07", "Engineering"),
    StockInfo("NSE:SALASAR-EQ",     "INE291I01014", "SALASAR",     "Salasar Techno Engineering",       "2017-07-24", "Engineering", "Listed Jul 2017"),
    StockInfo("NSE:SYRMA-EQ",         "INE0F4A01017", "SYRMA",         "Syrma SGS Technology Ltd",             "2022-08-26", "Engineering",      "Electronics mfg; listed Aug 2022"),
    StockInfo("NSE:STEELCAS-EQ",    "INE640A01031", "STEELCAS",    "Steel Strips Wheels Ltd",          "2000-05-18", "Auto"),
    StockInfo("NSE:GABRIEL-EQ",     "INE524A01029", "GABRIEL",     "Gabriel India Ltd",                "1995-03-01", "Auto"),
    StockInfo("NSE:JAMNAAUTO-EQ",   "INE226H01020", "JAMNAAUTO",   "Jamna Auto Industries Ltd",        "2000-06-13", "Auto"),
    StockInfo("NSE:ENDURANCE-EQ",   "INE913T01010", "ENDURANCE",   "Endurance Technologies Ltd",       "2016-10-19", "Auto"),
    StockInfo("NSE:DYNAMATECH-EQ",  "INE916P01024", "DYNAMATECH",  "Dynamatic Technologies Ltd",       "1994-09-15", "Engineering"),
    StockInfo("NSE:MINDACORP-EQ",       "INE261K01020", "MINDACORP",       "Minda Corporation Ltd",            "2011-06-28", "Auto"),
    StockInfo("NSE:SSWL-EQ",        "INE294I01015", "SSWL",        "Steel Strips Wheels Ltd",          "2000-05-18", "Auto"),
    StockInfo("NSE:SETCO-EQ",       "INE291G01016", "SETCO",       "Setco Automotive Ltd",             "2000-03-03", "Auto"),
]

# =============================================================================
# SECTOR 39 — GLASS, CERAMICS, BUILDING MATERIALS
# =============================================================================
BUILDING_MAT = [
    StockInfo("NSE:HINDZINC-EQ",    "INE267A01025", "HINDZINC",    "Hindustan Zinc Ltd",               "2002-11-22", "Metals"),
    StockInfo("NSE:SOMANYCERA-EQ",   "INE355F01017", "SOMANYCERA",    "Somany Ceramics Ltd",                   "2007-12-20", "Building",         "Pre-2017; full history; tiles mfg"),
    StockInfo("NSE:KAJARIACER-EQ",   "INE217B01036", "KAJARIACER",    "Kajaria Ceramics Ltd",                 "1999-03-17", "Building",         "Pre-2017; full history; tiles leader"),
    StockInfo("NSE:ASTRAL-EQ",      "INE006I01046", "ASTRAL",      "Astral Ltd",                       "2007-03-29", "Building"),
    StockInfo("NSE:PRINCEPIPE-EQ", "INE689W01016", "PRINCEPIPE", "Prince Pipes & Fittings Ltd",      "2019-12-30", "Building"),
    StockInfo("NSE:SUPREMEIND-EQ",  "INE195A01028", "SUPREMEIND",  "Supreme Industries Ltd",           "1995-03-01", "Building"),
    StockInfo("NSE:FINOLEXIND-EQ",     "INE516A01025", "FINOLEXIND",     "Finolex Industries Ltd",           "1995-03-01", "Building"),
    StockInfo("NSE:FINCABLES-EQ",     "INE514A01019", "FINCABLES",     "Finolex Cables Ltd",               "1995-03-01", "Engineering"),
    StockInfo("NSE:VGUARD-EQ",      "INE951I01027", "VGUARD",      "V-Guard Industries Ltd",           "2008-03-13", "Durables"),
    StockInfo("NSE:POLYPLEX-EQ",    "INE633A01013", "POLYPLEX",    "Polyplex Corporation Ltd",         "1995-03-01", "Chemicals"),
    StockInfo("NSE:NILKAMAL-EQ",    "INE310A01015", "NILKAMAL",    "Nilkamal Ltd",                     "1995-03-01", "Building"),
    StockInfo("NSE:CENTURYPLY-EQ",  "INE348B01021", "CENTURYPLY",  "Century Plyboards (India) Ltd",    "2004-12-16", "Building"),
    StockInfo("NSE:GREENPLY-EQ",    "INE461B01014", "GREENPLY",    "Greenply Industries Ltd",          "1998-06-15", "Building"),
    StockInfo("NSE:KSCL-EQ",        "INE848E01024", "KSCL",        "Kaveri Seed Company Ltd",          "2007-07-16", "Agro"),
]

# =============================================================================
# SECTOR 40 — JEWELLERY & WATCHES (ADDITIONAL UNIQUE)
# =============================================================================
JEWELLERY = [
    StockInfo("NSE:GOLDIAM-EQ",     "INE641F01014", "GOLDIAM",     "Goldiam International Ltd",        "1995-03-01", "Gems"),
    StockInfo("NSE:KDDL-EQ",        "INE788H01017", "KDDL",        "KDDL Ltd",                         "1994-09-15", "Gems"),
    StockInfo("NSE:TBZ-EQ",         "INE-0BU01014", "TBZ",         "Tribhovandas Bhimji Zaveri",       "2012-05-09", "Gems"),
]

# =============================================================================
# SECTOR 41 — FINAL FILL — QUALITY STOCKS WITH FULL HISTORY
# =============================================================================
FINAL_FILL = [
    StockInfo("NSE:HUDCO-EQ",       "INE031A01017", "HUDCO",       "Housing & Urban Dev Corp Ltd",     "2010-06-25", "Housing Finance"),
    StockInfo("NSE:NBCC-EQ",        "INE095N01031", "NBCC",        "NBCC (India) Ltd",                 "2012-04-12", "Infra"),   # dedup, will be dropped
    StockInfo("NSE:WABAG-EQ",       "INE975G01030", "WABAG",       "VA Tech Wabag Ltd",                "2010-09-22", "Engineering"),
    StockInfo("NSE:FINOLEXCAB-EQ",  "INE235A01022", "FINOLEXCAB",  "Finolex Cables Ltd",               "1995-03-01", "Engineering"),
    StockInfo("NSE:CAPACITE-EQ",    "INE0CB01014",  "CAPACITE",    "Capacit'e Infraprojects Ltd",      "2017-09-25", "Infra", "Listed Sep 2017"),
    StockInfo("NSE:ISGEC-EQ",       "INE411C01022", "ISGEC",       "ISGEC Heavy Engineering Ltd",      "1995-03-01", "Engineering"),  # dedup
    StockInfo("NSE:TANLA-EQ",       "INE483C01032", "TANLA",       "Tanla Platforms Ltd",              "2000-06-15", "IT"),   # dedup
    StockInfo("NSE:SYMPHONY-EQ",    "INE225C01021", "SYMPHONY",    "Symphony Ltd",                     "2010-09-14", "Durables"),
    StockInfo("NSE:VSTIND-EQ",      "INE302B01027", "VSTIND",      "VST Industries Ltd",               "1995-03-01", "FMCG"),   # dedup
    StockInfo("NSE:JSWHL-EQ",       "INE137O01025", "JSWHL",       "JSW Holdings Ltd",                 "2005-01-20", "Metals"),  # dedup
    StockInfo("NSE:KTKBANK-EQ",     "INE614B01018", "KTKBANK",     "Karnataka Bank Ltd",               "1998-01-05", "Banking"),
    StockInfo("NSE:JKBANK-EQ",        "INE168A01041", "JKBANK",        "Jammu & Kashmir Bank Ltd",             "1998-03-09", "Banking",          "Pre-2017; full history; private bank"),
    StockInfo("NSE:DCMSHRIRAM-EQ",  "INE843A01016", "DCMSHRIRAM",  "DCM Shriram Ltd",                  "1995-03-01", "Chemicals"),
    StockInfo("NSE:CHEMPLASTS-EQ",   "INE773A01014", "CHEMPLASTS",    "Chemplast Sanmar Ltd",                 "2021-08-24", "Chemicals",        "Specialty PVC; listed Aug 2021"),
    StockInfo("NSE:RAIN-EQ",        "INE855B01025", "RAIN",        "Rain Industries Ltd",              "1995-03-01", "Chemicals"),
    StockInfo("NSE:KALPATPOWR-EQ",   "INE220B01022", "KALPATPOWR",    "Kalpataru Power Transmission Ltd",     "2000-07-17", "Engineering",      "Pre-2017; full history; power T&D"),
    StockInfo("NSE:EVEREADY-EQ",    "INE128A01029", "EVEREADY",    "Eveready Industries India",        "1995-03-01", "Durables"),
    StockInfo("NSE:CENTURYTEX-EQ",  "INE055A01016", "CENTURYTEX",  "Century Textiles & Industries",    "1995-03-01", "Textiles"),
    StockInfo("NSE:ARMANFIN-EQ",      "INE109C01017", "ARMANFIN",      "Arman Financial Services Ltd",         "2014-04-07", "NBFC",             "Microfinance NBFC; pre-2017; full history"),
    StockInfo("NSE:NSLNISP-EQ",     "INE928A01036", "NSLNISP",     "NMDC Steel Ltd",                   "2023-02-09", "Metals", "Listed Feb 2023"),
    StockInfo("NSE:JINDALPOLY-EQ",  "INE785A01026", "JINDALPOLY",  "Jindal Poly Films Ltd",            "1995-03-01", "Chemicals"),
    StockInfo("NSE:CHOLAHLDNG-EQ",    "INE149A01033", "CHOLAHLDNG",    "Cholamandalam Financial Holdings Ltd",  "2001-04-12", "Capital Markets",  "Holding co; pre-2017; full history"),
]

# =============================================================================
# MASTER ASSEMBLY — DEDUPLICATION & FINAL 500
# =============================================================================

def _build_master_list() -> list[StockInfo]:
    """
    Combine all sector lists, deduplicate by ticker, 
    and return the final clean list.
    """
    all_stocks = (
        BANKING + NBFC + INSURANCE + CAPITAL_MARKETS + IT + ENERGY +
        POWER + AUTO + FMCG + RETAIL + PHARMA + METALS + CEMENT +
        INFRA + TELECOM + DEFENCE + AVIATION + ADANI + CHEMICALS +
        TEXTILES + ENGINEERING + MEDIA + DURABLES + HOSPITALS + MISC +
        HOUSING_FINANCE + MICROFINANCE + PAINTS + SUGAR_AGRO + PAPER +
        GEMS + IT_MID + NBFC2 + HEALTH2 + RAILWAYS + REALTY2 +
        CONGLOMERATES + POWER2 + BUILDING_MAT + JEWELLERY + FINAL_FILL
    )

    seen_tickers = set()
    deduplicated = []
    for s in all_stocks:
        if s.ticker not in seen_tickers:
            seen_tickers.add(s.ticker)
            deduplicated.append(s)

    return deduplicated


# The master list (all unique stocks, ~500)
NSE_500_STOCKS_INFO: list[StockInfo] = _build_master_list()

# Fyers-format symbols list (drop-in replacement for old NIFTY_50_STOCKS)
NSE_500_STOCKS: list[str] = [s.symbol for s in NSE_500_STOCKS_INFO]


# =============================================================================
# CONVENIENCE HELPERS — backward compatible with original API
# =============================================================================

# Index symbols
NIFTY_INDEX_SYMBOL = "NSE:NIFTY50-INDEX"

NIFTY_INDEX_ALTERNATIVES = [
    "NSE:NIFTY-INDEX",
    "NSE:NIFTY 50",
]


def get_stock_symbols() -> list[str]:
    """Returns list of NSE 500 stock symbols in Fyers format."""
    return NSE_500_STOCKS


def get_stock_info() -> list[StockInfo]:
    """Returns full StockInfo objects for all NSE 500 stocks."""
    return NSE_500_STOCKS_INFO


def get_index_symbol() -> str:
    """Returns Nifty 50 index symbol."""
    return NIFTY_INDEX_SYMBOL


def symbol_to_name(symbol: str) -> str:
    """
    Convert Fyers symbol to stock ticker name.
    Example: NSE:RELIANCE-EQ -> RELIANCE
    """
    return symbol.replace("NSE:", "").replace("-EQ", "")


def get_info_by_ticker(ticker: str) -> StockInfo | None:
    """Lookup full StockInfo by NSE ticker (e.g. 'RELIANCE')."""
    for s in NSE_500_STOCKS_INFO:
        if s.ticker == ticker:
            return s
    return None


def get_info_by_isin(isin: str) -> StockInfo | None:
    """Lookup StockInfo by ISIN."""
    for s in NSE_500_STOCKS_INFO:
        if s.isin == isin:
            return s
    return None


def get_stocks_by_sector(sector: str) -> list[StockInfo]:
    """Filter stocks by sector name (case-insensitive partial match)."""
    sector_lower = sector.lower()
    return [s for s in NSE_500_STOCKS_INFO if sector_lower in s.sector.lower()]


def get_full_history_stocks() -> list[StockInfo]:
    """
    Returns only stocks with FULL data coverage across BOTH windows:
      Jan 2017 – Dec 2019  AND  Jan 2022 – Feb 2026
    These are stocks listed before 1 January 2017 with no ticker-breaking events.
    """
    import datetime
    cutoff = datetime.date(2017, 1, 1)
    return [
        s for s in NSE_500_STOCKS_INFO
        if datetime.date.fromisoformat(s.listing_date) < cutoff
        and not s.notes  # no caveats
    ]


def get_partial_history_stocks() -> list[StockInfo]:
    """Stocks with partial coverage (listed 2017–2021 or have caveats)."""
    import datetime
    cutoff = datetime.date(2017, 1, 1)
    return [
        s for s in NSE_500_STOCKS_INFO
        if datetime.date.fromisoformat(s.listing_date) >= cutoff
        or s.notes
    ]


def validate_symbol(symbol: str) -> bool:
    """Validate if symbol is in the NSE 500 universe."""
    return symbol in NSE_500_STOCKS


def get_isin_map() -> dict[str, str]:
    """Returns {ticker: ISIN} mapping for all stocks."""
    return {s.ticker: s.isin for s in NSE_500_STOCKS_INFO}


def get_listing_date_map() -> dict[str, str]:
    """Returns {ticker: listing_date} mapping."""
    return {s.ticker: s.listing_date for s in NSE_500_STOCKS_INFO}


# =============================================================================
# KNOWN EXCLUSIONS — for documentation / audit trail
# =============================================================================
EXCLUDED_STOCKS = {
    "TATAMOTORS": "Demerger of CV business (TMLCV) effective 2025; price continuity broken",
    "JIOFIN":     "Listed Aug 2023; no 2017-2019 data",
    "PAYTM":      "Listed Nov 2021; no 2017-2019 data",
    "ZOMATO":     "Listed Jul 2021; renamed ETERNAL; no 2017-2019 data",
    "NYKAA":      "Listed Nov 2021 (included but flagged); no 2017-2019 data",
    "LIC":        "Listed May 2022; no 2017-2019 data",
    "DELHIVERY":  "Listed May 2022; no 2017-2019 data",
    "POLICYBAZAAR":"Listed Nov 2021 (included but flagged)",
    "VODAIDEA":   "Extreme financial distress; near-delisting risk",
    "SWIGGY":     "Listed Nov 2024; no historical data",
    "JIOFINANCE": "Same as JIOFIN",
}


# =============================================================================
# MAIN — diagnostic output
# =============================================================================
if __name__ == "__main__":
    import datetime

    all_stocks = NSE_500_STOCKS_INFO
    full_hist = get_full_history_stocks()
    partial = get_partial_history_stocks()

    print("=" * 70)
    print("NSE 500 STOCK UNIVERSE — PRICE PREDICTION PROJECT")
    print("=" * 70)
    print(f"Total unique stocks  : {len(all_stocks)}")
    print(f"Full history (pre-2017 listing, no caveats): {len(full_hist)}")
    print(f"Partial / flagged    : {len(partial)}")
    print()

    # Sector breakdown
    from collections import Counter
    sector_counts = Counter(s.sector for s in all_stocks)
    print("Sector Breakdown:")
    for sector, count in sorted(sector_counts.items(), key=lambda x: -x[1]):
        print(f"  {sector:<30s} {count:>3d} stocks")
    print()

    print("Sample (first 15 stocks):")
    print(f"  {'#':>3}  {'Ticker':<15} {'Name':<40} {'Listed':<12} {'Sector'}")
    print("  " + "-" * 90)
    for i, s in enumerate(all_stocks[:15], 1):
        print(f"  {i:>3}. {s.ticker:<15} {s.name:<40} {s.listing_date:<12} {s.sector}")

    print()
    print("ISIN map sample:")
    isin_map = get_isin_map()
    for ticker, isin in list(isin_map.items())[:5]:
        print(f"  {ticker:<15} → {isin}")

    print()
    print(f"Index symbol: {NIFTY_INDEX_SYMBOL}")
    print()
    print("Known exclusions:")
    for ticker, reason in EXCLUDED_STOCKS.items():
        print(f"  {ticker:<15} : {reason}")
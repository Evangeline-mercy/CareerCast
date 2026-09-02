import os
import pandas as pd

INPUT_FILE = "it_jobs_required_skills.csv"
OUTPUT_DIR = "results/milestone2_unified"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "career_profiles_96.csv"
)

print("=" * 90)
print("CAREERCAST MILESTONE 2")
print("CAREER TAXONOMY EXPANSION")
print("=" * 90)

# ---------------------------------------------------------
# 1. Load existing 87 career profiles
# ---------------------------------------------------------
print("\n[1] Loading existing career profiles...")

df = pd.read_csv(INPUT_FILE)

print("Existing careers :", len(df))
print("Columns          :", list(df.columns))

# ---------------------------------------------------------
# 2. Additional career profiles required by validation
# ---------------------------------------------------------
additional_profiles = [

    {
        "job_title": "VLSI Engineer",
        "required_skills":
            "Verilog | SystemVerilog | VHDL | RTL Design | Digital Electronics | "
            "ASIC Design | FPGA | CMOS | Semiconductor Design | Logic Synthesis | "
            "Static Timing Analysis | Physical Design | Floorplanning | Placement | "
            "Routing | Clock Tree Synthesis | DFT | Cadence | Synopsys | "
            "Design Verification"
    },

    {
        "job_title": "Electronics Engineer",
        "required_skills":
            "Digital Electronics | Analog Electronics | Circuit Design | PCB Design | "
            "Microcontrollers | Embedded Systems | Embedded C | Sensors | "
            "Signal Processing | Communication Systems | Power Electronics | "
            "Oscilloscopes | Multimeter | MATLAB | SPICE | Soldering | "
            "Testing and Debugging | Electronics Prototyping | EMI/EMC | "
            "Technical Documentation"
    },

    {
        "job_title": "Firmware Engineer",
        "required_skills":
            "C | C++ | Embedded C | Microcontrollers | ARM | RTOS | Firmware Development | "
            "Device Drivers | UART | SPI | I2C | CAN | GPIO | Interrupts | Timers | "
            "Memory Management | Debugging | Git | Embedded Linux | Hardware Interfaces"
    },

    {
        "job_title": "IoT Engineer",
        "required_skills":
            "IoT | Embedded Systems | Arduino | ESP32 | Raspberry Pi | Embedded C | "
            "Python | Sensors | MQTT | HTTP | REST API | Wireless Communication | "
            "WiFi | Bluetooth | Cloud Computing | AWS IoT | Data Processing | "
            "Edge Computing | Linux | IoT Security"
    },

    {
        "job_title": "Signal Processing Engineer",
        "required_skills":
            "Digital Signal Processing | MATLAB | Python | Fourier Transform | FFT | "
            "Filtering | FIR Filters | IIR Filters | Sampling | Modulation | "
            "Signal Analysis | Time Frequency Analysis | Spectral Analysis | "
            "Image Processing | Audio Processing | Communication Systems | "
            "Statistics | Linear Algebra | Numerical Methods | DSP Algorithms"
    },

    {
        "job_title": "Electrical Engineer",
        "required_skills":
            "Electrical Circuits | Power Systems | Electrical Machines | Power Electronics | "
            "Control Systems | MATLAB | Simulink | Circuit Analysis | Transformers | "
            "Generators | Motors | PLC | SCADA | Electrical Wiring | AutoCAD Electrical | "
            "Protection Systems | Renewable Energy | Instrumentation | "
            "Electrical Safety | Testing and Maintenance"
    },

    {
        "job_title": "Computer Hardware Engineer",
        "required_skills":
            "Computer Architecture | Digital Electronics | Microprocessors | Microcontrollers | "
            "Embedded Systems | PCB Design | Circuit Design | FPGA | Verilog | "
            "Hardware Debugging | Memory Systems | CPU Architecture | Bus Architecture | "
            "PCIe | USB | I2C | SPI | UART | Linux | Hardware Testing"
    },

    {
        "job_title": "Computer and Information Research Scientist",
        "required_skills":
            "Python | Machine Learning | Deep Learning | Artificial Intelligence | "
            "Algorithms | Data Structures | Mathematics | Statistics | Research Methods | "
            "Computer Vision | NLP | Neural Networks | Optimization | "
            "Scientific Computing | TensorFlow | PyTorch | Experiment Design | "
            "Data Analysis | Academic Research | Technical Writing"
    },

    {
        "job_title": "Operations Research Analyst",
        "required_skills":
            "Operations Research | Mathematics | Statistics | Optimization | "
            "Linear Programming | Integer Programming | Probability | Simulation | "
            "Python | R | SQL | Data Analysis | Predictive Modeling | "
            "Decision Analysis | Forecasting | Mathematical Modeling | "
            "Excel | Tableau | Solver | Supply Chain Analytics"
    }
]

# ---------------------------------------------------------
# 3. Validate the new profiles
# ---------------------------------------------------------
print("\n[2] Validating additional career profiles...")

for profile in additional_profiles:
    skills = [
        x.strip()
        for x in profile["required_skills"].split("|")
        if x.strip()
    ]

    if len(skills) != 20:
        raise ValueError(
            f"{profile['job_title']} has {len(skills)} skills instead of 20."
        )

print(
    "Additional profiles :",
    len(additional_profiles)
)

# ---------------------------------------------------------
# 4. Check duplicates
# ---------------------------------------------------------
existing_titles = set(
    df["job_title"].astype(str).str.strip()
)

duplicate_titles = [
    p["job_title"]
    for p in additional_profiles
    if p["job_title"] in existing_titles
]

if duplicate_titles:
    print("\nWARNING: These careers already exist:")
    for title in duplicate_titles:
        print(" -", title)

    additional_profiles = [
        p for p in additional_profiles
        if p["job_title"] not in existing_titles
    ]

# ---------------------------------------------------------
# 5. Add profiles
# ---------------------------------------------------------
print("\n[3] Adding new career profiles...")

new_df = pd.DataFrame(additional_profiles)

final_df = pd.concat(
    [df, new_df],
    ignore_index=True
)

# Remove accidental duplicate job titles
final_df = final_df.drop_duplicates(
    subset=["job_title"],
    keep="first"
)

# ---------------------------------------------------------
# 6. Final validation
# ---------------------------------------------------------
print("\n" + "=" * 90)
print("FINAL CAREER PROFILE QUALITY CHECK")
print("=" * 90)

skill_counts = (
    final_df["required_skills"]
    .fillna("")
    .str.split("|")
    .str.len()
)

print("Total careers :", len(final_df))
print("Unique careers:", final_df["job_title"].nunique())
print("Minimum skills:", skill_counts.min())
print("Maximum skills:", skill_counts.max())
print("Average skills:", skill_counts.mean())

invalid = final_df[
    (skill_counts != 20)
]

print(
    "\nProfiles with skill count != 20 :",
    len(invalid)
)

if len(invalid) > 0:
    print("\nINVALID PROFILES:")
    print(
        invalid[["job_title", "required_skills"]]
        .to_string(index=False)
    )
    raise ValueError(
        "Some career profiles do not contain exactly 20 skills."
    )

# ---------------------------------------------------------
# 7. Save
# ---------------------------------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

final_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 90)
print("FINAL DATASET")
print("=" * 90)

print("Rows       :", len(final_df))
print("Careers    :", final_df["job_title"].nunique())
print("Skills/profile: 20")

print("\nNew careers added:")

for profile in additional_profiles:
    print(" -", profile["job_title"])

print("\nSaved:")
print(OUTPUT_FILE)

print("\n" + "=" * 90)
print("CAREER TAXONOMY EXPANSION COMPLETE")
print("=" * 90)
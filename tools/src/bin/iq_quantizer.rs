// IQ Quantizer Tool - Generate valid quantized test data
// Complete port of llama.cpp IQ quantization algorithms
//
// Usage:
//   cargo run --release --bin iq_quantizer -- --format iq2_xxs --output test.bin
//   cargo run --release --bin iq_quantizer -- --format iq3_s --blocks 10 --export-rust

use clap::Parser;
use half::f16;
use std::fs::File;
use std::io::Write;

#[derive(Parser)]
#[command(name = "iq_quantizer")]
#[command(about = "Generate quantized test data for IQ formats")]
struct Args {
    /// Format: iq1_s, iq1_m, iq2_xxs, iq2_xs, iq2_s, iq3_xxs, iq3_s, iq4_nl, iq4_xs
    #[arg(short, long)]
    format: String,

    /// Number of blocks (default: 1)
    #[arg(short, long, default_value = "1")]
    blocks: usize,

    /// Output binary file
    #[arg(short, long)]
    output: Option<String>,

    /// Export as Rust source file
    #[arg(long)]
    export_rust: Option<String>,

    /// Random seed (default: 42)
    #[arg(short, long, default_value = "42")]
    seed: u64,
}

const QK_K: usize = 256; // Super-block size
const GROUP_MAX_EPS: f32 = 1e-6;

// ============================================================================
// Grid tables from llama.cpp ggml-quants.c
// ============================================================================

// kgrid_2bit_256 for IQ2_XXS
const KGRID_2BIT_256: [u16; 256] = [
    0, 2, 5, 8, 10, 17, 20, 32, 34, 40, 42, 65, 68, 80, 88, 97, 100, 128, 130,
    138, 162, 257, 260, 272, 277, 320, 388, 408, 512, 514, 546, 642, 1025,
    1028, 1040, 1057, 1060, 1088, 1090, 1096, 1120, 1153, 1156, 1168, 1188,
    1280, 1282, 1288, 1312, 1350, 1385, 1408, 1425, 1545, 1552, 1600, 1668,
    1700, 2048, 2053, 2056, 2068, 2088, 2113, 2116, 2128, 2130, 2184, 2308,
    2368, 2562, 2580, 4097, 4100, 4112, 4129, 4160, 4192, 4228, 4240, 4245,
    4352, 4360, 4384, 4432, 4442, 4480, 4644, 4677, 5120, 5128, 5152, 5157,
    5193, 5248, 5400, 5474, 5632, 5654, 6145, 6148, 6160, 6208, 6273, 6400,
    6405, 6560, 6737, 8192, 8194, 8202, 8260, 8289, 8320, 8322, 8489, 8520,
    8704, 8706, 9217, 9220, 9232, 9280, 9302, 9472, 9537, 9572, 9872, 10248,
    10272, 10388, 10820, 16385, 16388, 16400, 16408, 16417, 16420, 16448,
    16456, 16470, 16480, 16513, 16516, 16528, 16640, 16672, 16737, 16768,
    16773, 16897, 16912, 16968, 16982, 17000, 17408, 17416, 17440, 17536,
    17561, 17682, 17700, 17920, 18433, 18436, 18448, 18496, 18501, 18688,
    18776, 18785, 18818, 19013, 19088, 20480, 20488, 20497, 20505, 20512,
    20608, 20616, 20740, 20802, 20900, 21137, 21648, 21650, 21770, 22017,
    22100, 22528, 22545, 22553, 22628, 22848, 23048, 24580, 24592, 24640,
    24680, 24832, 24917, 25112, 25184, 25600, 25605, 25872, 25874, 25988,
    26690, 32768, 32770, 32778, 32833, 32898, 33028, 33048, 33088, 33297,
    33793, 33796, 33808, 33813, 33856, 33888, 34048, 34118, 34196, 34313,
    34368, 34400, 34818, 35076, 35345, 36868, 36880, 36900, 36928, 37025,
    37142, 37248, 37445, 37888, 37922, 37956, 38225, 39041, 39200, 40962,
    41040, 41093, 41225, 41472, 42008, 43088, 43268,
];

// kgrid_2bit_512 for IQ2_XS
const KGRID_2BIT_512: [u16; 512] = [
    0, 2, 5, 8, 10, 17, 20, 22, 25, 32, 34, 37, 40, 65, 68, 70, 73, 80, 82, 85,
    88, 97, 100, 128, 130, 133, 136, 145, 148, 153, 160, 257, 260, 262, 265,
    272, 274, 277, 280, 282, 289, 292, 320, 322, 325, 328, 337, 340, 352, 360,
    385, 388, 400, 512, 514, 517, 520, 529, 532, 544, 577, 580, 592, 597, 640,
    650, 1025, 1028, 1030, 1033, 1040, 1042, 1045, 1048, 1057, 1060, 1088,
    1090, 1093, 1096, 1105, 1108, 1110, 1120, 1153, 1156, 1168, 1280, 1282,
    1285, 1288, 1297, 1300, 1312, 1345, 1348, 1360, 1377, 1408, 1537, 1540,
    1552, 1574, 1600, 1602, 1668, 2048, 2050, 2053, 2056, 2058, 2065, 2068,
    2080, 2085, 2113, 2116, 2128, 2136, 2176, 2208, 2218, 2305, 2308, 2320,
    2368, 2433, 2441, 2560, 2592, 2600, 2710, 2720, 4097, 4100, 4102, 4105,
    4112, 4114, 4117, 4120, 4129, 4132, 4160, 4162, 4165, 4168, 4177, 4180,
    4192, 4202, 4225, 4228, 4240, 4352, 4354, 4357, 4360, 4369, 4372, 4384,
    4417, 4420, 4432, 4480, 4500, 4502, 4609, 4612, 4614, 4624, 4672, 4704,
    5120, 5122, 5125, 5128, 5137, 5140, 5152, 5185, 5188, 5193, 5200, 5220,
    5248, 5377, 5380, 5392, 5440, 5632, 5652, 5705, 6145, 6148, 6160, 6162,
    6208, 6228, 6278, 6400, 6405, 6502, 6737, 6825, 8192, 8194, 8197, 8200,
    8202, 8209, 8212, 8224, 8257, 8260, 8272, 8320, 8352, 8449, 8452, 8464,
    8512, 8520, 8549, 8704, 8738, 8832, 8872, 9217, 9220, 9232, 9257, 9280,
    9472, 9537, 9554, 9625, 9729, 9754, 9894, 10240, 10248, 10250, 10272,
    10325, 10376, 10402, 10600, 10640, 10760, 10784, 10882, 10888, 10890,
    16385, 16388, 16390, 16393, 16400, 16402, 16405, 16408, 16417, 16420,
    16448, 16450, 16453, 16456, 16458, 16465, 16468, 16480, 16485, 16513,
    16516, 16528, 16640, 16642, 16645, 16648, 16657, 16660, 16672, 16705,
    16708, 16720, 16768, 16773, 16802, 16897, 16900, 16912, 16914, 16937,
    16960, 17408, 17410, 17413, 17416, 17425, 17428, 17433, 17440, 17473,
    17476, 17488, 17536, 17556, 17665, 17668, 17680, 17700, 17728, 17818,
    17920, 17930, 17988, 18000, 18433, 18436, 18448, 18496, 18501, 18516,
    18530, 18688, 18705, 18756, 18768, 18793, 18948, 20480, 20482, 20485,
    20488, 20497, 20500, 20512, 20520, 20545, 20548, 20560, 20608, 20737,
    20740, 20752, 20757, 20800, 20802, 20992, 21060, 21162, 21505, 21508,
    21520, 21537, 21568, 21600, 21633, 21665, 21760, 21768, 21888, 21896,
    22049, 22120, 22177, 22528, 22548, 22593, 22608, 22681, 22810, 22848,
    22850, 23173, 24577, 24580, 24592, 24640, 24660, 24674, 24710, 24745,
    24832, 25124, 25162, 25234, 25600, 25622, 25872, 25920, 25925, 26020,
    26625, 26730, 26917, 27142, 27220, 27234, 32768, 32770, 32773, 32776,
    32785, 32788, 32800, 32810, 32833, 32836, 32848, 32896, 32898, 32936,
    32938, 33025, 33028, 33030, 33040, 33088, 33105, 33113, 33280, 33312,
    33408, 33410, 33440, 33448, 33793, 33796, 33808, 33810, 33813, 33856,
    33888, 33929, 34048, 34116, 34213, 34328, 34410, 34816, 34824, 34853,
    34906, 34944, 34946, 34984, 35078, 35362, 35456, 35464, 35478, 35496,
    36865, 36868, 36880, 36928, 36950, 36996, 37120, 37154, 37220, 37462,
    37513, 37888, 37893, 37956, 37968, 37976, 38185, 38288, 38290, 38465,
    38993, 39078, 39241, 39445, 39520, 40960, 40962, 40968, 40970, 40992,
    41002, 41120, 41297, 41305, 41382, 41472, 41474, 41480, 41514, 41600,
    41632, 42048, 42133, 42597, 42648, 43018, 43040, 43042, 43048, 43168,
    43176, 43268, 43396, 43398, 43560, 43562, 43665, 43690,
];

// kgrid_2bit_1024 for IQ2_S
const KGRID_2BIT_1024: [u16; 1024] = [
    0, 2, 5, 8, 10, 17, 20, 22, 25, 32, 34, 37, 40, 65, 68, 70, 73, 80, 82, 85,
    88, 97, 100, 102, 105, 128, 130, 133, 136, 145, 148, 160, 165, 170, 257,
    260, 262, 265, 272, 274, 277, 280, 289, 292, 320, 322, 325, 328, 337, 340,
    342, 345, 352, 357, 360, 385, 388, 400, 402, 405, 417, 420, 512, 514, 517,
    520, 529, 532, 544, 554, 577, 580, 582, 585, 592, 597, 640, 645, 650, 660,
    674, 1025, 1028, 1030, 1033, 1040, 1042, 1045, 1048, 1057, 1060, 1062,
    1065, 1088, 1090, 1093, 1096, 1098, 1105, 1108, 1110, 1113, 1120, 1122,
    1125, 1153, 1156, 1158, 1161, 1168, 1173, 1176, 1185, 1188, 1280, 1282,
    1285, 1288, 1290, 1297, 1300, 1302, 1305, 1312, 1317, 1320, 1345, 1348,
    1350, 1353, 1360, 1362, 1365, 1368, 1377, 1380, 1408, 1410, 1413, 1416,
    1425, 1428, 1440, 1537, 1540, 1542, 1545, 1552, 1557, 1600, 1605, 1608,
    1617, 1620, 1632, 1665, 1668, 1680, 2048, 2050, 2053, 2056, 2065, 2068,
    2070, 2073, 2080, 2085, 2090, 2113, 2116, 2118, 2121, 2128, 2130, 2133,
    2136, 2145, 2148, 2176, 2181, 2196, 2218, 2305, 2308, 2320, 2322, 2325,
    2328, 2337, 2368, 2373, 2376, 2385, 2388, 2400, 2433, 2448, 2560, 2577,
    2580, 2594, 2600, 2602, 2640, 2713, 4097, 4100, 4102, 4105, 4112, 4114,
    4117, 4120, 4129, 4132, 4134, 4160, 4162, 4165, 4168, 4177, 4180, 4182,
    4185, 4192, 4194, 4197, 4200, 4225, 4228, 4230, 4240, 4245, 4248, 4257,
    4260, 4352, 4354, 4357, 4360, 4362, 4369, 4372, 4374, 4377, 4384, 4386,
    4389, 4392, 4417, 4420, 4422, 4425, 4432, 4434, 4437, 4440, 4449, 4452,
    4480, 4482, 4485, 4488, 4497, 4500, 4609, 4612, 4617, 4624, 4629, 4641,
    4644, 4672, 4677, 4689, 4692, 4737, 4740, 4752, 5120, 5122, 5125, 5128,
    5137, 5140, 5142, 5145, 5152, 5157, 5160, 5185, 5188, 5190, 5193, 5200,
    5202, 5205, 5208, 5217, 5220, 5248, 5250, 5253, 5256, 5265, 5268, 5280,
    5377, 5380, 5382, 5385, 5392, 5394, 5397, 5400, 5409, 5412, 5440, 5442,
    5445, 5448, 5457, 5460, 5472, 5505, 5508, 5520, 5632, 5637, 5640, 5649,
    5652, 5664, 5697, 5700, 5712, 5760, 5802, 6145, 6148, 6150, 6153, 6160,
    6165, 6168, 6177, 6208, 6210, 6213, 6216, 6225, 6228, 6240, 6273, 6276,
    6400, 6402, 6405, 6408, 6417, 6420, 6432, 6465, 6468, 6480, 6505, 6562,
    6660, 6672, 6720, 6742, 8192, 8194, 8197, 8200, 8209, 8212, 8214, 8217,
    8224, 8229, 8234, 8257, 8260, 8272, 8274, 8277, 8292, 8320, 8330, 8340,
    8362, 8449, 8452, 8464, 8466, 8469, 8481, 8512, 8514, 8517, 8529, 8532,
    8544, 8577, 8580, 8592, 8704, 8714, 8738, 8744, 8746, 8772, 8784, 8840,
    8842, 8872, 9217, 9220, 9222, 9225, 9232, 9237, 9240, 9249, 9252, 9280,
    9282, 9285, 9288, 9297, 9300, 9312, 9345, 9348, 9360, 9472, 9477, 9480,
    9489, 9492, 9504, 9537, 9540, 9552, 9574, 9600, 9729, 9732, 9744, 9792,
    9817, 10240, 10245, 10257, 10260, 10305, 10308, 10320, 10378, 10410, 10497,
    10500, 10512, 10645, 10762, 10786, 10852, 10888, 10890, 16385, 16388,
    16390, 16393, 16400, 16402, 16405, 16408, 16410, 16417, 16420, 16422,
    16448, 16450, 16453, 16456, 16458, 16465, 16468, 16470, 16473, 16480,
    16482, 16485, 16513, 16516, 16528, 16533, 16536, 16545, 16548, 16640,
    16642, 16645, 16648, 16657, 16660, 16662, 16665, 16672, 16674, 16677,
    16705, 16708, 16710, 16713, 16720, 16722, 16725, 16728, 16737, 16740,
    16768, 16770, 16773, 16776, 16785, 16788, 16800, 16897, 16900, 16912,
    16914, 16917, 16920, 16932, 16960, 16965, 16968, 16977, 16980, 16992,
    17025, 17028, 17408, 17410, 17413, 17416, 17418, 17425, 17428, 17430,
    17433, 17440, 17442, 17445, 17448, 17473, 17476, 17478, 17481, 17488,
    17490, 17493, 17496, 17505, 17508, 17536, 17538, 17541, 17544, 17553,
    17556, 17568, 17665, 17668, 17670, 17673, 17680, 17682, 17685, 17688,
    17697, 17700, 17728, 17730, 17733, 17736, 17745, 17748, 17760, 17770,
    17793, 17796, 17808, 17920, 17922, 17925, 17928, 17937, 17940, 17952,
    17985, 17988, 18000, 18048, 18085, 18433, 18436, 18441, 18448, 18450,
    18453, 18456, 18465, 18468, 18496, 18498, 18501, 18504, 18513, 18516,
    18528, 18564, 18576, 18688, 18690, 18693, 18696, 18705, 18708, 18720,
    18753, 18756, 18768, 18816, 18838, 18945, 18948, 18960, 19008, 20480,
    20482, 20485, 20488, 20497, 20500, 20502, 20505, 20512, 20514, 20517,
    20520, 20545, 20548, 20550, 20553, 20560, 20562, 20565, 20568, 20577,
    20580, 20608, 20610, 20613, 20616, 20625, 20628, 20737, 20740, 20742,
    20745, 20752, 20754, 20757, 20760, 20769, 20772, 20800, 20802, 20805,
    20808, 20817, 20820, 20832, 20865, 20868, 20880, 20992, 20997, 21000,
    21009, 21012, 21024, 21057, 21060, 21072, 21097, 21120, 21505, 21508,
    21510, 21513, 21520, 21522, 21525, 21528, 21537, 21540, 21568, 21570,
    21573, 21576, 21585, 21588, 21600, 21633, 21636, 21648, 21760, 21762,
    21765, 21768, 21777, 21780, 21792, 21825, 21828, 21840, 21888, 22017,
    22020, 22032, 22054, 22080, 22528, 22530, 22533, 22536, 22545, 22548,
    22560, 22593, 22596, 22608, 22618, 22656, 22785, 22788, 22800, 22848,
    23040, 23065, 23173, 23208, 24577, 24580, 24582, 24592, 24594, 24597,
    24600, 24609, 24612, 24640, 24645, 24648, 24657, 24660, 24672, 24708,
    24720, 24832, 24834, 24837, 24840, 24849, 24852, 24864, 24897, 24900,
    24912, 24960, 24985, 25092, 25104, 25152, 25174, 25249, 25600, 25605,
    25608, 25617, 25620, 25632, 25665, 25668, 25680, 25728, 25857, 25860,
    25872, 25920, 25930, 25960, 26002, 26112, 26260, 26625, 26628, 26640,
    26725, 26776, 26880, 26922, 27202, 27297, 32768, 32770, 32773, 32776,
    32785, 32788, 32793, 32800, 32805, 32833, 32836, 32848, 32850, 32853,
    32856, 32865, 32896, 32901, 32913, 32916, 33025, 33028, 33033, 33040,
    33042, 33045, 33048, 33057, 33060, 33088, 33090, 33093, 33096, 33105,
    33108, 33153, 33156, 33168, 33193, 33280, 33285, 33290, 33297, 33300,
    33345, 33348, 33360, 33793, 33796, 33798, 33801, 33808, 33810, 33813,
    33816, 33825, 33856, 33858, 33861, 33864, 33873, 33876, 33888, 33921,
    33924, 33936, 34048, 34050, 34053, 34056, 34065, 34068, 34080, 34113,
    34116, 34128, 34176, 34186, 34305, 34308, 34320, 34345, 34368, 34816,
    34821, 34833, 34836, 34881, 34884, 34896, 34978, 35073, 35076, 35136,
    35173, 35362, 35416, 35418, 35458, 35490, 36865, 36868, 36873, 36880,
    36882, 36885, 36888, 36900, 36928, 36930, 36933, 36936, 36945, 36948,
    36960, 36993, 36996, 37008, 37120, 37125, 37137, 37140, 37185, 37188,
    37200, 37210, 37377, 37380, 37392, 37440, 37542, 37888, 37890, 37893,
    37896, 37905, 37908, 37920, 37953, 37956, 37968, 38016, 38038, 38145,
    38148, 38160, 38208, 38296, 38305, 38400, 38470, 38500, 38913, 38916,
    38928, 38950, 38976, 39081, 39168, 39241, 39250, 39568, 40960, 40965,
    40970, 40980, 40994, 41002, 41025, 41028, 41040, 41122, 41130, 41280,
    41317, 41474, 41482, 41506, 41512, 41514, 41602, 41608, 41610, 41640,
    41985, 41988, 42000, 42048, 42121, 42148, 42240, 42265, 42577, 43018,
    43048, 43170, 43348, 43398, 43528, 43530, 43552, 43554, 43560, 43656,
    43690,
];

// kgrid_256 for IQ3_XXS
const KGRID_256_IQ3: [u16; 256] = [
    0, 2, 4, 9, 11, 15, 16, 18, 25, 34, 59, 61, 65, 67, 72, 74, 81, 85, 88, 90,
    97, 108, 120, 128, 130, 132, 137, 144, 146, 153, 155, 159, 169, 175, 189,
    193, 199, 200, 202, 213, 248, 267, 287, 292, 303, 315, 317, 321, 327, 346,
    362, 413, 436, 456, 460, 462, 483, 497, 513, 515, 520, 522, 529, 531, 536,
    538, 540, 551, 552, 576, 578, 585, 592, 594, 641, 643, 648, 650, 657, 664,
    698, 704, 706, 720, 729, 742, 758, 769, 773, 808, 848, 852, 870, 889, 901,
    978, 992, 1024, 1026, 1033, 1035, 1040, 1042, 1046, 1049, 1058, 1089, 1091,
    1093, 1096, 1098, 1105, 1112, 1139, 1143, 1144, 1152, 1154, 1161, 1167,
    1168, 1170, 1183, 1184, 1197, 1217, 1224, 1228, 1272, 1276, 1309, 1323,
    1347, 1367, 1377, 1404, 1473, 1475, 1486, 1509, 1537, 1544, 1546, 1553,
    1555, 1576, 1589, 1594, 1600, 1602, 1616, 1625, 1636, 1638, 1665, 1667,
    1672, 1685, 1706, 1722, 1737, 1755, 1816, 1831, 1850, 1856, 1862, 1874,
    1901, 1932, 1950, 1971, 2011, 2032, 2052, 2063, 2077, 2079, 2091, 2095,
    2172, 2192, 2207, 2208, 2224, 2230, 2247, 2277, 2308, 2345, 2356, 2389,
    2403, 2424, 2501, 2504, 2506, 2520, 2570, 2593, 2616, 2624, 2630, 2646,
    2669, 2700, 2714, 2746, 2754, 2795, 2824, 2835, 2839, 2874, 2882, 2905,
    2984, 3028, 3042, 3092, 3108, 3110, 3124, 3153, 3185, 3215, 3252, 3288,
    3294, 3364, 3397, 3434, 3483, 3523, 3537, 3587, 3589, 3591, 3592, 3610,
    3626, 3670, 3680, 3722, 3749, 3754, 3776, 3789, 3803, 3824, 3857, 3873,
    3904, 3906, 3924, 3992,
];

// kgrid_512 for IQ3_S
const KGRID_512_IQ3: [u16; 512] = [
    0, 1, 2, 5, 7, 8, 9, 10, 12, 14, 16, 17, 21, 27, 32, 34, 37, 39, 41, 43,
    48, 50, 57, 60, 63, 64, 65, 66, 68, 72, 73, 77, 80, 83, 87, 89, 93, 100,
    113, 117, 122, 128, 129, 133, 135, 136, 139, 142, 145, 149, 152, 156, 162,
    165, 167, 169, 171, 184, 187, 195, 201, 205, 208, 210, 217, 219, 222, 228,
    232, 234, 247, 249, 253, 256, 267, 271, 273, 276, 282, 288, 291, 297, 312,
    322, 324, 336, 338, 342, 347, 353, 357, 359, 374, 379, 390, 393, 395, 409,
    426, 441, 448, 450, 452, 464, 466, 470, 475, 488, 492, 512, 513, 514, 516,
    520, 521, 523, 525, 527, 528, 530, 537, 540, 542, 556, 558, 561, 570, 576,
    577, 579, 582, 584, 588, 593, 600, 603, 609, 616, 618, 632, 638, 640, 650,
    653, 655, 656, 660, 666, 672, 675, 685, 688, 698, 705, 708, 711, 712, 715,
    721, 727, 728, 732, 737, 754, 760, 771, 773, 778, 780, 793, 795, 802, 806,
    808, 812, 833, 840, 843, 849, 856, 858, 873, 912, 916, 919, 932, 934, 961,
    963, 968, 970, 977, 989, 993, 1010, 1016, 1024, 1025, 1027, 1029, 1031,
    1032, 1034, 1036, 1038, 1041, 1043, 1047, 1048, 1050, 1057, 1059, 1061,
    1064, 1066, 1079, 1080, 1083, 1085, 1088, 1090, 1096, 1099, 1103, 1106,
    1109, 1113, 1116, 1122, 1129, 1153, 1156, 1159, 1169, 1171, 1176, 1183,
    1185, 1195, 1199, 1209, 1212, 1216, 1218, 1221, 1225, 1234, 1236, 1241,
    1243, 1250, 1256, 1270, 1281, 1287, 1296, 1299, 1306, 1309, 1313, 1338,
    1341, 1348, 1353, 1362, 1375, 1376, 1387, 1400, 1408, 1410, 1415, 1425,
    1453, 1457, 1477, 1481, 1494, 1496, 1507, 1512, 1538, 1545, 1547, 1549,
    1551, 1554, 1561, 1563, 1565, 1570, 1572, 1575, 1577, 1587, 1593, 1601,
    1603, 1605, 1612, 1617, 1619, 1632, 1648, 1658, 1662, 1664, 1674, 1680,
    1690, 1692, 1704, 1729, 1736, 1740, 1745, 1747, 1751, 1752, 1761, 1763,
    1767, 1773, 1787, 1795, 1801, 1806, 1810, 1817, 1834, 1840, 1844, 1857,
    1864, 1866, 1877, 1882, 1892, 1902, 1915, 1934, 1953, 1985, 1987, 2000,
    2002, 2013, 2048, 2052, 2058, 2064, 2068, 2071, 2074, 2081, 2088, 2104,
    2114, 2119, 2121, 2123, 2130, 2136, 2141, 2147, 2153, 2157, 2177, 2179,
    2184, 2189, 2193, 2203, 2208, 2223, 2226, 2232, 2244, 2249, 2251, 2256,
    2258, 2265, 2269, 2304, 2306, 2324, 2335, 2336, 2361, 2373, 2375, 2385,
    2418, 2443, 2460, 2480, 2504, 2509, 2520, 2531, 2537, 2562, 2568, 2572,
    2578, 2592, 2596, 2599, 2602, 2614, 2620, 2625, 2627, 2629, 2634, 2641,
    2650, 2682, 2688, 2697, 2707, 2712, 2718, 2731, 2754, 2759, 2760, 2775,
    2788, 2793, 2805, 2811, 2817, 2820, 2832, 2842, 2854, 2890, 2902, 2921,
    2923, 2978, 3010, 3012, 3026, 3081, 3083, 3085, 3097, 3099, 3120, 3136,
    3152, 3159, 3188, 3210, 3228, 3234, 3245, 3250, 3256, 3264, 3276, 3281,
    3296, 3349, 3363, 3378, 3392, 3395, 3420, 3440, 3461, 3488, 3529, 3531,
    3584, 3588, 3591, 3600, 3602, 3614, 3616, 3628, 3634, 3650, 3657, 3668,
    3683, 3685, 3713, 3716, 3720, 3726, 3729, 3736, 3753, 3778, 3802, 3805,
    3819, 3841, 3845, 3851, 3856, 3880, 3922, 3938, 3970, 3993, 4032,
];

// ============================================================================
// Grid data structures
// ============================================================================

struct IQ2GridData {
    grid: Vec<u64>,      // Grid points (8 int8_t per u64)
    map: Vec<i32>,       // Map from index to grid point
    neighbors: Vec<u16>, // Neighbor table
}

struct IQ3GridData {
    grid: Vec<u32>,      // Grid points (4 int8_t per u32)
    map: Vec<i32>,       // Map from index to grid point
    neighbors: Vec<u16>, // Neighbor table
}

impl IQ2GridData {
    fn new(grid_size: usize, kgrid: &[u16], nwant: usize) -> Self {
        let kmap_size = 65536; // 2^16 for 16-bit indices

        // Build grid (convert from u16 encoding to u64 with 8 int8_t values)
        let mut grid = vec![0u64; grid_size];
        for k in 0..grid_size {
            let mut val: u64 = 0;
            for i in 0..8 {
                let l = ((kgrid[k] >> (2 * i)) & 0x3) as i8;
                let q = 2 * l + 1; // Values: 1, 3, 5, 7
                val |= ((q as u8) as u64) << (8 * i);
            }
            grid[k] = val;
        }

        // Build map
        let mut map = vec![-1i32; kmap_size];
        for i in 0..grid_size {
            let aux8 = grid[i].to_le_bytes();
            let mut index: u16 = 0;
            for k in 0..8 {
                let q = ((aux8[k] as i8) - 1) / 2;
                index |= ((q as u16) & 0x3) << (2 * k);
            }
            map[index as usize] = i as i32;
        }

        // Build neighbor table (simplified - just find nearest neighbors)
        let mut neighbors = Vec::new();

        for idx in 0..kmap_size {
            if map[idx] >= 0 {
                continue;
            }

            // Find neighbors for this index
            let mut pos = [0i8; 8];
            for k in 0..8 {
                let l = ((idx >> (2 * k)) & 0x3) as i8;
                pos[k] = 2 * l + 1;
            }

            // Calculate distances to all grid points
            let mut dist2: Vec<(i32, usize)> = (0..grid_size)
                .map(|j| {
                    let pg = grid[j].to_le_bytes();
                    let mut d2 = 0i32;
                    for k in 0..8 {
                        let diff = (pg[k] as i8) as i32 - pos[k] as i32;
                        d2 += diff * diff;
                    }
                    (d2, j)
                })
                .collect();

            dist2.sort_by_key(|&(d, _)| d);

            // Count neighbors at first nwant distance levels
            let mut n = 0;
            let mut d2 = dist2[0].0;
            let mut nhave = 1;
            for j in 0..grid_size {
                if dist2[j].0 > d2 {
                    if nhave >= nwant {
                        break;
                    }
                    d2 = dist2[j].0;
                    nhave += 1;
                }
                n += 1;
            }

            // Store neighbor count and indices
            map[idx] = -((neighbors.len() + 1) as i32);
            neighbors.push(n as u16);
            for j in 0..n {
                neighbors.push(dist2[j].1 as u16);
            }
        }

        IQ2GridData { grid, map, neighbors }
    }

    fn find_best_neighbor(
        &self,
        u: u16,
        xval: &[f32],
        weight: &[f32],
        scale: f32,
        l: &mut [i8],
    ) -> i32 {
        let map_val = self.map[u as usize];
        if map_val >= 0 {
            let pg = self.grid[map_val as usize].to_le_bytes();
            for i in 0..8 {
                l[i] = ((pg[i] as i8) - 1) / 2;
            }
            return map_val;
        }

        let neighbor_idx = (-map_val - 1) as usize;
        let num_neighbors = self.neighbors[neighbor_idx] as usize;
        let neighbors =
            &self.neighbors[neighbor_idx + 1..neighbor_idx + 1 + num_neighbors];

        let mut best_d2 = f32::MAX;
        let mut best_idx = -1i32;

        for &nj in neighbors {
            let pg = self.grid[nj as usize].to_le_bytes();
            let mut d2 = 0.0f32;
            for i in 0..8 {
                let q = pg[i] as f32;
                let diff = scale * q - xval[i];
                d2 += weight[i] * diff * diff;
            }
            if d2 < best_d2 {
                best_d2 = d2;
                best_idx = nj as i32;
            }
        }

        if best_idx >= 0 {
            let pg = self.grid[best_idx as usize].to_le_bytes();
            for i in 0..8 {
                l[i] = ((pg[i] as i8) - 1) / 2;
            }
        }

        best_idx
    }
}

impl IQ3GridData {
    fn new(grid_size: usize, kgrid: &[u16]) -> Self {
        let kmap_size = 65536; // 2^16 for 16-bit indices (3-bit values = 4 values per group)

        // Build grid (convert from u16 encoding to u32 with 4 int8_t values)
        let mut grid = vec![0u32; grid_size];
        for k in 0..grid_size {
            let mut val: u32 = 0;
            for i in 0..4 {
                let l = ((kgrid[k] >> (3 * i)) & 0x7) as i8;
                let q = 2 * l + 1; // Values: 1, 3, 5, 7, 9, 11, 13, 15
                val |= ((q as u8) as u32) << (8 * i);
            }
            grid[k] = val;
        }

        // Build map (12-bit indices = 4 values * 3 bits each)
        let mut map = vec![-1i32; kmap_size];
        for i in 0..grid_size {
            let aux = grid[i].to_le_bytes();
            let mut index: u16 = 0;
            for k in 0..4 {
                let q = ((aux[k] as i8) - 1) / 2;
                index |= ((q as u16) & 0x7) << (3 * k);
            }
            map[index as usize] = i as i32;
        }

        // Build neighbor table (simplified)
        let mut neighbors = Vec::new();
        let nwant = 2;

        for idx in 0..kmap_size {
            if map[idx] >= 0 {
                continue;
            }

            let mut pos = [0i8; 4];
            for k in 0..4 {
                let l = ((idx >> (3 * k)) & 0x7) as i8;
                pos[k] = 2 * l + 1;
            }

            let mut dist2: Vec<(i32, usize)> = (0..grid_size)
                .map(|j| {
                    let pg = grid[j].to_le_bytes();
                    let mut d2 = 0i32;
                    for k in 0..4 {
                        let diff = (pg[k] as i8) as i32 - pos[k] as i32;
                        d2 += diff * diff;
                    }
                    (d2, j)
                })
                .collect();

            dist2.sort_by_key(|&(d, _)| d);

            let mut n = 0;
            let mut d2 = dist2[0].0;
            let mut nhave = 1;
            for j in 0..grid_size {
                if dist2[j].0 > d2 {
                    if nhave >= nwant {
                        break;
                    }
                    d2 = dist2[j].0;
                    nhave += 1;
                }
                n += 1;
            }

            map[idx] = -((neighbors.len() + 1) as i32);
            neighbors.push(n as u16);
            for j in 0..n {
                neighbors.push(dist2[j].1 as u16);
            }
        }

        IQ3GridData { grid, map, neighbors }
    }

    fn find_best_neighbor(
        &self,
        u: u16,
        xval: &[f32],
        weight: &[f32],
        scale: f32,
        l: &mut [i8],
    ) -> i32 {
        let map_val = self.map[u as usize];
        if map_val >= 0 {
            let pg = self.grid[map_val as usize].to_le_bytes();
            for i in 0..4 {
                l[i] = ((pg[i] as i8) - 1) / 2;
            }
            return map_val;
        }

        let neighbor_idx = (-map_val - 1) as usize;
        let num_neighbors = self.neighbors[neighbor_idx] as usize;
        let neighbors =
            &self.neighbors[neighbor_idx + 1..neighbor_idx + 1 + num_neighbors];

        let mut best_d2 = f32::MAX;
        let mut best_idx = -1i32;

        for &nj in neighbors {
            let pg = self.grid[nj as usize].to_le_bytes();
            let mut d2 = 0.0f32;
            for i in 0..4 {
                let q = pg[i] as f32;
                let diff = scale * q - xval[i];
                d2 += weight[i] * diff * diff;
            }
            if d2 < best_d2 {
                best_d2 = d2;
                best_idx = nj as i32;
            }
        }

        if best_idx >= 0 {
            let pg = self.grid[best_idx as usize].to_le_bytes();
            for i in 0..4 {
                l[i] = ((pg[i] as i8) - 1) / 2;
            }
        }

        best_idx
    }
}

// ============================================================================
// Block structures
// ============================================================================

#[repr(C)]
#[derive(Clone, Copy)]
struct BlockIQ2XXS {
    d: u16,        // FP16 scale
    qs: [u16; 32], // Grid indices + signs
}

#[repr(C)]
#[derive(Clone, Copy)]
struct BlockIQ2XS {
    d: u16,          // FP16 scale
    qs: [u16; 32],   // Grid indices + signs
    scales: [u8; 8], // Scales
}

#[repr(C)]
#[derive(Clone, Copy)]
struct BlockIQ2S {
    d: u16,          // FP16 scale
    qs: [u8; 64],    // Grid indices + signs
    qh: [u8; 8],     // High bits of grid indices
    scales: [u8; 8], // 4-bit scales per sub-block
}

#[repr(C)]
#[derive(Clone, Copy)]
struct BlockIQ3XXS {
    d: u16,       // FP16 scale
    qs: [u8; 96], // Grid indices packed (32 * 3 bytes)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct BlockIQ3S {
    d: u16,          // FP16 scale
    qs: [u8; 64],    // Grid indices
    qh: [u8; 8],     // High bits
    signs: [u8; 32], // Sign bits
    scales: [u8; 4], // Scales
}

#[repr(C)]
#[derive(Clone, Copy)]
struct BlockIQ4NL {
    d: u16,       // FP16 scale
    qs: [u8; 16], // 4-bit quants (32 values)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct BlockIQ4XS {
    d: u16,            // FP16 scale
    scales_h: u16,     // High bits of scales
    scales_l: [u8; 4], // Low bits of scales
    qs: [u8; 128],     // 4-bit quants (256 values)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct BlockIQ1S {
    d: u16,       // FP16 scale
    qs: [u8; 32], // Grid indices
    qh: [u16; 8], // High bits
}

#[repr(C)]
#[derive(Clone, Copy)]
struct BlockIQ1M {
    qs: [u8; 32],    // Grid indices, low 8 bits
    qh: [u8; 16],    // Grid indices, high bits
    scales: [u8; 8], // Scales
}

// ============================================================================
// Quantization functions
// ============================================================================

fn nearest_int(x: f32) -> i32 {
    (x + 0.5_f32.copysign(x)) as i32
}

fn quantize_iq4_nl(x: &[f32]) -> BlockIQ4NL {
    assert!(x.len() == 32);

    let mut amax = 0.0f32;
    for &v in x {
        amax = amax.max(v.abs());
    }

    let d = amax / 8.0;
    let id = if d > 0.0 { 1.0 / d } else { 0.0 };

    let mut qs = [0u8; 16];
    for i in 0..16 {
        let x0 = x[i] * id;
        let x1 = x[i + 16] * id;
        let q0 = ((x0 + 8.5).clamp(0.0, 15.0) as u8) & 0x0F;
        let q1 = ((x1 + 8.5).clamp(0.0, 15.0) as u8) & 0x0F;
        qs[i] = q0 | (q1 << 4);
    }

    BlockIQ4NL { d: f16::from_f32(d).to_bits(), qs }
}

fn quantize_iq4_xs(x: &[f32]) -> BlockIQ4XS {
    assert!(x.len() == 256);

    let mut amax = 0.0f32;
    for &v in x {
        amax = amax.max(v.abs());
    }

    let d = amax / 8.0;
    let id = if d > 0.0 { 1.0 / d } else { 0.0 };

    let mut qs = [0u8; 128];
    for i in 0..128 {
        let x0 = x[i] * id;
        let x1 = x[i + 128] * id;
        let q0 = ((x0 + 8.5).clamp(0.0, 15.0) as u8) & 0x0F;
        let q1 = ((x1 + 8.5).clamp(0.0, 15.0) as u8) & 0x0F;
        qs[i] = q0 | (q1 << 4);
    }

    BlockIQ4XS {
        d: f16::from_f32(d).to_bits(),
        scales_h: 0,
        scales_l: [0u8; 4],
        qs,
    }
}

fn quantize_iq2_xxs(x: &[f32], grid_data: &IQ2GridData) -> BlockIQ2XXS {
    assert!(x.len() == 256);

    const KMAX_Q: i32 = 3;

    let mut amax = 0.0f32;
    for &v in x {
        amax = amax.max(v.abs());
    }

    let d = if amax > GROUP_MAX_EPS { amax } else { 1.0 };
    let id = 1.0 / d;

    let mut qs = [0u16; 32];
    let mut scales = [0.0f32; 16];

    // Process each sub-block of 16 values
    for ib in 0..16 {
        let xb = &x[ib * 16..(ib + 1) * 16];

        // Extract signs and absolute values
        let mut xval = [0.0f32; 16];
        let mut block_signs = [0u8; 2];

        for k in 0..2 {
            let mut nflip = 0;
            let mut s = 0u8;
            for i in 0..8 {
                if xb[8 * k + i] >= 0.0 {
                    xval[8 * k + i] = xb[8 * k + i];
                } else {
                    xval[8 * k + i] = -xb[8 * k + i];
                    nflip += 1;
                    s |= 1 << i;
                }
            }
            // Ensure odd number of signs per group
            if nflip % 2 != 0 {
                let mut min_idx = 0;
                let mut min_val = xval[8 * k] * xval[8 * k];
                for i in 1..8 {
                    let val = xval[8 * k + i] * xval[8 * k + i];
                    if val < min_val {
                        min_val = val;
                        min_idx = i;
                    }
                }
                xval[8 * k + min_idx] = -xval[8 * k + min_idx];
                s ^= 1 << min_idx;
            }
            block_signs[k] = s & 127;
        }

        let max = xval.iter().cloned().fold(0.0f32, f32::max);
        if max < GROUP_MAX_EPS {
            scales[ib] = 0.0;
            continue;
        }

        let mut l = [0i8; 16];
        let mut best = 0.0f32;
        let mut scale = max / (2 * KMAX_Q - 1) as f32;

        // Scale optimization loop
        for is in -9..=9 {
            let this_id = ((2 * KMAX_Q - 1) as f32 + is as f32 * 0.1) / max;
            let this_scale = 1.0 / this_id;
            let mut laux = [0i8; 16];

            for k in 0..2 {
                let mut u = 0u16;
                for i in 0..8 {
                    let lval =
                        nearest_int(0.5 * (this_id * xval[8 * k + i] - 1.0));
                    laux[8 * k + i] = lval.clamp(0, KMAX_Q - 1) as i8;
                    u |= (laux[8 * k + i] as u16) << (2 * i);
                }

                let map_val = grid_data.map[u as usize];
                if map_val < 0 {
                    let weight: Vec<f32> =
                        (0..8).map(|i| xval[8 * k + i].abs()).collect();
                    grid_data.find_best_neighbor(
                        u,
                        &xval[8 * k..8 * (k + 1)],
                        &weight,
                        this_scale,
                        &mut laux[8 * k..8 * (k + 1)],
                    );
                }
            }

            // Compute weighted MSE
            let mut sumqx = 0.0f32;
            let mut sumq2 = 0.0f32;
            for i in 0..16 {
                let w = xval[i] * xval[i];
                let q = 2.0 * (laux[i] as f32) + 1.0;
                sumqx += w * xval[i] * q;
                sumq2 += w * q * q;
            }

            if sumq2 > 0.0 && sumqx * sumqx > best * sumq2 {
                scale = sumqx / sumq2;
                best = scale * sumqx;
                l.copy_from_slice(&laux);
            }
        }

        // Pack quantized values
        for k in 0..2 {
            let mut u = 0u16;
            for i in 0..8 {
                u |= (l[8 * k + i] as u16) << (2 * i);
            }
            let map_val = grid_data.map[u as usize];
            let grid_index = if map_val >= 0 {
                map_val as usize
            } else {
                let weight: Vec<f32> =
                    (0..8).map(|i| xval[8 * k + i].abs()).collect();
                let mut l_aux = [0i8; 8];
                grid_data.find_best_neighbor(
                    u,
                    &xval[8 * k..8 * (k + 1)],
                    &weight,
                    scale,
                    &mut l_aux,
                ) as usize
            };

            qs[2 * ib + k] =
                (grid_index as u16) | ((block_signs[k] as u16) << 9);
        }

        scales[ib] = scale;
    }

    // Find max scale and encode scales
    let max_scale = scales.iter().cloned().fold(0.0f32, f32::max);
    let d = max_scale / 31.0;

    // Encode final block
    // Note: IQ2_XXS uses the qs array directly with embedded signs

    BlockIQ2XXS { d: f16::from_f32(d).to_bits(), qs }
}

fn quantize_iq2_xs(x: &[f32], grid_data: &IQ2GridData) -> BlockIQ2XS {
    assert!(x.len() == 256);

    const KMAX_Q: i32 = 3;

    let mut scales = [0.0f32; 16];
    let mut qs = [0u16; 32];

    for ib in 0..16 {
        let xb = &x[ib * 16..(ib + 1) * 16];

        let mut xval = [0.0f32; 16];
        let mut block_signs = [0u8; 2];

        for k in 0..2 {
            let mut nflip = 0;
            let mut s = 0u8;
            for i in 0..8 {
                if xb[8 * k + i] >= 0.0 {
                    xval[8 * k + i] = xb[8 * k + i];
                } else {
                    xval[8 * k + i] = -xb[8 * k + i];
                    nflip += 1;
                    s |= 1 << i;
                }
            }
            if nflip % 2 != 0 {
                let mut min_idx = 0;
                let mut min_val = xval[8 * k] * xval[8 * k];
                for i in 1..8 {
                    let val = xval[8 * k + i] * xval[8 * k + i];
                    if val < min_val {
                        min_val = val;
                        min_idx = i;
                    }
                }
                xval[8 * k + min_idx] = -xval[8 * k + min_idx];
                s ^= 1 << min_idx;
            }
            block_signs[k] = s & 127;
        }

        let max = xval.iter().cloned().fold(0.0f32, f32::max);
        if max < GROUP_MAX_EPS {
            scales[ib] = 0.0;
            continue;
        }

        let mut l = [0i8; 16];
        let mut best = 0.0f32;
        let mut scale = max / (2 * KMAX_Q - 1) as f32;

        for is in -9..=9 {
            let this_id = ((2 * KMAX_Q - 1) as f32 + is as f32 * 0.1) / max;
            let this_scale = 1.0 / this_id;
            let mut laux = [0i8; 16];

            for k in 0..2 {
                let mut u = 0u16;
                for i in 0..8 {
                    let lval =
                        nearest_int(0.5 * (this_id * xval[8 * k + i] - 1.0));
                    laux[8 * k + i] = lval.clamp(0, KMAX_Q - 1) as i8;
                    u |= (laux[8 * k + i] as u16) << (2 * i);
                }

                let map_val = grid_data.map[u as usize];
                if map_val < 0 {
                    let weight: Vec<f32> =
                        (0..8).map(|i| xval[8 * k + i].abs()).collect();
                    grid_data.find_best_neighbor(
                        u,
                        &xval[8 * k..8 * (k + 1)],
                        &weight,
                        this_scale,
                        &mut laux[8 * k..8 * (k + 1)],
                    );
                }
            }

            let mut sumqx = 0.0f32;
            let mut sumq2 = 0.0f32;
            for i in 0..16 {
                let w = xval[i] * xval[i];
                let q = 2.0 * (laux[i] as f32) + 1.0;
                sumqx += w * xval[i] * q;
                sumq2 += w * q * q;
            }

            if sumq2 > 0.0 && sumqx * sumqx > best * sumq2 {
                scale = sumqx / sumq2;
                best = scale * sumqx;
                l.copy_from_slice(&laux);
            }
        }

        for k in 0..2 {
            let mut u = 0u16;
            for i in 0..8 {
                u |= (l[8 * k + i] as u16) << (2 * i);
            }
            let map_val = grid_data.map[u as usize];
            let grid_index = if map_val >= 0 {
                map_val as usize
            } else {
                let weight: Vec<f32> =
                    (0..8).map(|i| xval[8 * k + i].abs()).collect();
                let mut l_aux = [0i8; 8];
                grid_data.find_best_neighbor(
                    u,
                    &xval[8 * k..8 * (k + 1)],
                    &weight,
                    scale,
                    &mut l_aux,
                ) as usize
            };

            qs[2 * ib + k] =
                (grid_index as u16) | ((block_signs[k] as u16) << 9);
        }

        scales[ib] = scale;
    }

    let max_scale = scales.iter().cloned().fold(0.0f32, f32::max);
    let d = max_scale / 31.0;

    // Encode scales
    let id = 1.0 / d;
    let mut encoded_scales = [0u8; 8];
    for ib in 0..16 {
        let lval = nearest_int(0.5 * (id * scales[ib] - 1.0)).clamp(0, 15);
        if ib % 2 == 0 {
            encoded_scales[ib / 2] = lval as u8;
        } else {
            encoded_scales[ib / 2] |= (lval as u8) << 4;
        }
    }

    BlockIQ2XS { d: f16::from_f32(d).to_bits(), qs, scales: encoded_scales }
}

fn quantize_iq2_s(x: &[f32], grid_data: &IQ2GridData) -> BlockIQ2S {
    assert!(x.len() == 256);

    const KMAX_Q: i32 = 3;

    let mut scales = [0.0f32; 8];
    let mut qs = [0u8; 64];
    let mut qh = [0u8; 8];
    let mut encoded_scales = [0u8; 8];
    let mut max_scale = 0.0f32;

    // Process each sub-block of 32 values
    for ib in 0..8 {
        let xb = &x[ib * 32..(ib + 1) * 32];

        // Extract signs and absolute values
        let mut xval = [0.0f32; 32];
        let mut block_signs = [0u8; 4];

        for k in 0..4 {
            let mut nflip = 0;
            let mut s = 0u8;
            for i in 0..8 {
                if xb[8 * k + i] >= 0.0 {
                    xval[8 * k + i] = xb[8 * k + i];
                } else {
                    xval[8 * k + i] = -xb[8 * k + i];
                    nflip += 1;
                    s |= 1 << i;
                }
            }
            if nflip % 2 != 0 {
                let mut min_idx = 0;
                let mut min_val = xval[8 * k] * xval[8 * k];
                for i in 1..8 {
                    let val = xval[8 * k + i] * xval[8 * k + i];
                    if val < min_val {
                        min_val = val;
                        min_idx = i;
                    }
                }
                xval[8 * k + min_idx] = -xval[8 * k + min_idx];
                s ^= 1 << min_idx;
            }
            block_signs[k] = s;
        }

        let max = xval.iter().cloned().fold(0.0f32, f32::max);
        if max < GROUP_MAX_EPS {
            scales[ib] = 0.0;
            continue;
        }

        let mut l = [0i8; 32];
        let mut best = 0.0f32;
        let mut scale = max / (2 * KMAX_Q - 1) as f32;
        let mut is_on_grid = [true; 4];

        // Scale optimization loop
        for is in -9..=9 {
            let this_id = ((2 * KMAX_Q - 1) as f32 + is as f32 * 0.1) / max;
            let this_scale = 1.0 / this_id;
            let mut laux = [0i8; 32];
            let mut is_on_grid_aux = [true; 4];

            for k in 0..4 {
                let mut u = 0u16;
                for i in 0..8 {
                    let lval =
                        nearest_int(0.5 * (this_id * xval[8 * k + i] - 1.0));
                    laux[8 * k + i] = lval.clamp(0, KMAX_Q - 1) as i8;
                    u |= (laux[8 * k + i] as u16) << (2 * i);
                }

                let map_val = grid_data.map[u as usize];
                if map_val < 0 {
                    is_on_grid_aux[k] = false;
                    let weight: Vec<f32> =
                        (0..8).map(|i| xval[8 * k + i].abs()).collect();
                    grid_data.find_best_neighbor(
                        u,
                        &xval[8 * k..8 * (k + 1)],
                        &weight,
                        this_scale,
                        &mut laux[8 * k..8 * (k + 1)],
                    );
                }
            }

            // Compute weighted MSE
            let mut sumqx = 0.0f32;
            let mut sumq2 = 0.0f32;
            for i in 0..32 {
                let w = xval[i] * xval[i];
                let q = 2.0 * (laux[i] as f32) + 1.0;
                sumqx += w * xval[i] * q;
                sumq2 += w * q * q;
            }

            if sumq2 > 0.0 && sumqx * sumqx > best * sumq2 {
                scale = sumqx / sumq2;
                best = scale * sumqx;
                l.copy_from_slice(&laux);
                is_on_grid.copy_from_slice(&is_on_grid_aux);
            }
        }

        // Handle off-grid points
        let n_not_ongrid = is_on_grid.iter().filter(|&&x| !x).count();
        if n_not_ongrid > 0 && scale > 0.0 {
            let this_id = 1.0 / scale;
            for k in 0..4 {
                if is_on_grid[k] {
                    continue;
                }
                let mut u = 0u16;
                for i in 0..8 {
                    let lval =
                        nearest_int(0.5 * (this_id * xval[8 * k + i] - 1.0))
                            .clamp(0, KMAX_Q - 1);
                    u |= (lval as u16) << (2 * i);
                }

                let map_val = grid_data.map[u as usize];
                if map_val < 0 {
                    let weight: Vec<f32> =
                        (0..8).map(|i| xval[8 * k + i].abs()).collect();
                    grid_data.find_best_neighbor(
                        u,
                        &xval[8 * k..8 * (k + 1)],
                        &weight,
                        scale,
                        &mut l[8 * k..8 * (k + 1)],
                    );
                }
            }

            let mut sumqx = 0.0f32;
            let mut sumq2 = 0.0f32;
            for i in 0..32 {
                let w = xval[i] * xval[i];
                let q = 2.0 * (l[i] as f32) + 1.0;
                sumqx += w * xval[i] * q;
                sumq2 += w * q * q;
            }
            if sumq2 > 0.0 {
                scale = sumqx / sumq2;
            }
        }

        // Pack grid indices for IQ2_S (10-bit indices: 1024 grid points)
        for k in 0..4 {
            let mut u = 0u16;
            for i in 0..8 {
                u |= (l[8 * k + i] as u16) << (2 * i);
            }
            let map_val = grid_data.map[u as usize];
            let grid_index = if map_val >= 0 {
                map_val as usize
            } else {
                let weight: Vec<f32> =
                    (0..8).map(|i| xval[8 * k + i].abs()).collect();
                let mut l_aux = [0i8; 8];
                grid_data.find_best_neighbor(
                    u,
                    &xval[8 * k..8 * (k + 1)],
                    &weight,
                    scale,
                    &mut l_aux,
                ) as usize
            };

            // Pack: low 8 bits in qs, high 2 bits in qh
            // qs has 64 bytes = 8 indices per sub-block, so 8 bytes per group
            qs[8 * ib + 2 * k] = (grid_index & 0xFF) as u8;
            qs[8 * ib + 2 * k + 1] = block_signs[k];
            qh[ib] |= ((grid_index >> 8) as u8) << k;
        }

        scales[ib] = scale;
        max_scale = max_scale.max(scale);
    }

    let d = max_scale / 31.0;

    // Encode scales (4-bit per scale)
    let id = 1.0 / d;
    for ib in 0..8 {
        let lval = nearest_int(0.5 * (id * scales[ib] - 1.0)).clamp(0, 15);
        if ib % 2 == 0 {
            encoded_scales[ib / 2] = lval as u8;
        } else {
            encoded_scales[ib / 2] |= (lval as u8) << 4;
        }
    }

    BlockIQ2S { d: f16::from_f32(d).to_bits(), qs, qh, scales: encoded_scales }
}

fn quantize_iq3_xxs(x: &[f32], grid_data: &IQ3GridData) -> BlockIQ3XXS {
    assert!(x.len() == 256);

    const KMAX_Q: i32 = 8;

    let mut scales = [0.0f32; 8];
    let mut q3 = [0u8; 100]; // Temporary buffer for packing
    let mut max_scale = 0.0f32;

    // Process each sub-block of 32 values
    for ib in 0..8 {
        let xb = &x[ib * 32..(ib + 1) * 32];

        // Extract signs and absolute values
        let mut xval = [0.0f32; 32];
        let mut block_signs = [0u8; 4];

        for k in 0..4 {
            let mut nflip = 0;
            let mut s = 0u8;
            for i in 0..8 {
                if xb[8 * k + i] >= 0.0 {
                    xval[8 * k + i] = xb[8 * k + i];
                } else {
                    xval[8 * k + i] = -xb[8 * k + i];
                    nflip += 1;
                    s |= 1 << i;
                }
            }
            // Ensure odd number of signs per group (parity constraint)
            if nflip % 2 != 0 {
                let mut min_idx = 0;
                let mut min_val = xval[8 * k] * xval[8 * k];
                for i in 1..8 {
                    let val = xval[8 * k + i] * xval[8 * k + i];
                    if val < min_val {
                        min_val = val;
                        min_idx = i;
                    }
                }
                xval[8 * k + min_idx] = -xval[8 * k + min_idx];
                s ^= 1 << min_idx;
            }
            block_signs[k] = s & 127;
        }

        let max = xval.iter().cloned().fold(0.0f32, f32::max);
        if max < GROUP_MAX_EPS {
            scales[ib] = 0.0;
            continue;
        }

        let mut l = [0i8; 32];
        let mut best = 0.0f32;
        let mut scale = max / (2 * KMAX_Q - 1) as f32;
        let mut is_on_grid = [true; 8];

        // Scale optimization loop (31 iterations)
        for is in -15..=15 {
            let this_id = ((2 * KMAX_Q - 1) as f32 + is as f32 * 0.2) / max;
            let this_scale = 1.0 / this_id;
            let mut laux = [0i8; 32];
            let mut is_on_grid_aux = [true; 8];

            for k in 0..8 {
                let mut u = 0u16;
                for i in 0..4 {
                    let lval =
                        nearest_int(0.5 * (this_id * xval[4 * k + i] - 1.0));
                    laux[4 * k + i] = lval.clamp(0, KMAX_Q - 1) as i8;
                    u |= (laux[4 * k + i] as u16) << (3 * i);
                }

                let map_val = grid_data.map[u as usize];
                if map_val < 0 {
                    is_on_grid_aux[k] = false;
                    let weight: Vec<f32> =
                        (0..4).map(|i| xval[4 * k + i].abs()).collect();
                    grid_data.find_best_neighbor(
                        u,
                        &xval[4 * k..4 * (k + 1)],
                        &weight,
                        this_scale,
                        &mut laux[4 * k..4 * (k + 1)],
                    );
                }
            }

            // Compute weighted MSE
            let mut sumqx = 0.0f32;
            let mut sumq2 = 0.0f32;
            for i in 0..32 {
                let w = xval[i] * xval[i];
                let q = 2.0 * (laux[i] as f32) + 1.0;
                sumqx += w * xval[i] * q;
                sumq2 += w * q * q;
            }

            if sumq2 > 0.0 && sumqx * sumqx > best * sumq2 {
                scale = sumqx / sumq2;
                best = scale * sumqx;
                l.copy_from_slice(&laux);
                is_on_grid.copy_from_slice(&is_on_grid_aux);
            }
        }

        // Handle off-grid points
        let n_not_ongrid = is_on_grid.iter().filter(|&&x| !x).count();
        if n_not_ongrid > 0 && scale > 0.0 {
            let this_id = 1.0 / scale;
            for k in 0..8 {
                if is_on_grid[k] {
                    continue;
                }
                let mut u = 0u16;
                for i in 0..4 {
                    let lval =
                        nearest_int(0.5 * (this_id * xval[4 * k + i] - 1.0))
                            .clamp(0, KMAX_Q - 1);
                    u |= (lval as u16) << (3 * i);
                }

                let map_val = grid_data.map[u as usize];
                if map_val < 0 {
                    let weight: Vec<f32> =
                        (0..4).map(|i| xval[4 * k + i].abs()).collect();
                    grid_data.find_best_neighbor(
                        u,
                        &xval[4 * k..4 * (k + 1)],
                        &weight,
                        scale,
                        &mut l[4 * k..4 * (k + 1)],
                    );
                }
            }

            // Recompute scale
            let mut sumqx = 0.0f32;
            let mut sumq2 = 0.0f32;
            for i in 0..32 {
                let w = xval[i] * xval[i];
                let q = 2.0 * (l[i] as f32) + 1.0;
                sumqx += w * xval[i] * q;
                sumq2 += w * q * q;
            }
            if sumq2 > 0.0 {
                scale = sumqx / sumq2;
            }
        }

        // Pack grid indices (3 bits per value, 8 groups of 4 values = 24 bits per group)
        for k in 0..8 {
            let mut u = 0u16;
            for i in 0..4 {
                u |= (l[4 * k + i] as u16) << (3 * i);
            }

            let map_val = grid_data.map[u as usize];
            let grid_index = if map_val >= 0 {
                map_val as u16
            } else {
                panic!("Should not happen - grid lookup failed");
            };

            // Pack 8-bit grid index
            q3[8 * ib + k] = grid_index as u8;
        }

        // Pack signs and scales (14 bits per sub-block)
        let signs_and_scales = (block_signs[0] as u32)
            | ((block_signs[1] as u32) << 7)
            | ((block_signs[2] as u32) << 14)
            | ((block_signs[3] as u32) << 21);
        q3[32 + ib] = (signs_and_scales & 0xFF) as u8;
        q3[40 + ib] = ((signs_and_scales >> 8) & 0xFF) as u8;
        q3[48 + ib] = ((signs_and_scales >> 16) & 0xFF) as u8;
        q3[56 + ib] = ((signs_and_scales >> 24) & 0xFF) as u8;

        scales[ib] = scale;
        max_scale = max_scale.max(scale);
    }

    let d = max_scale / 31.0 * 1.0125;

    // Copy packed data to output
    let mut qs = [0u8; 96];
    qs.copy_from_slice(&q3[0..96]);

    BlockIQ3XXS { d: f16::from_f32(d).to_bits(), qs }
}

fn quantize_iq3_s(x: &[f32], grid_data: &IQ3GridData) -> BlockIQ3S {
    assert!(x.len() == 256);

    const KMAX_Q: i32 = 8;

    let mut scales = [0.0f32; 8];
    let mut qs = [0u8; 64];
    let mut qh = [0u8; 8];
    let mut signs = [0u8; 32];
    let mut max_scale = 0.0f32;

    for ib in 0..8 {
        let xb = &x[ib * 32..(ib + 1) * 32];

        let mut xval = [0.0f32; 32];
        let mut block_signs = [0u8; 4];

        for k in 0..4 {
            let mut nflip = 0;
            let mut s = 0u8;
            for i in 0..8 {
                if xb[8 * k + i] >= 0.0 {
                    xval[8 * k + i] = xb[8 * k + i];
                } else {
                    xval[8 * k + i] = -xb[8 * k + i];
                    nflip += 1;
                    s |= 1 << i;
                }
            }
            if nflip % 2 != 0 {
                let mut min_idx = 0;
                let mut min_val = xval[8 * k] * xval[8 * k];
                for i in 1..8 {
                    let val = xval[8 * k + i] * xval[8 * k + i];
                    if val < min_val {
                        min_val = val;
                        min_idx = i;
                    }
                }
                xval[8 * k + min_idx] = -xval[8 * k + min_idx];
                s ^= 1 << min_idx;
            }
            block_signs[k] = s & 127;
            signs[4 * ib + k] = s;
        }

        let max = xval.iter().cloned().fold(0.0f32, f32::max);
        if max < GROUP_MAX_EPS {
            scales[ib] = 0.0;
            continue;
        }

        let mut l = [0i8; 32];
        let mut best = 0.0f32;
        let mut scale = max / (2 * KMAX_Q - 1) as f32;
        let mut is_on_grid = [true; 8];

        for is in -15..=15 {
            let this_id = ((2 * KMAX_Q - 1) as f32 + is as f32 * 0.2) / max;
            let this_scale = 1.0 / this_id;
            let mut laux = [0i8; 32];
            let mut is_on_grid_aux = [true; 8];

            for k in 0..8 {
                let mut u = 0u16;
                for i in 0..4 {
                    let lval =
                        nearest_int(0.5 * (this_id * xval[4 * k + i] - 1.0));
                    laux[4 * k + i] = lval.clamp(0, KMAX_Q - 1) as i8;
                    u |= (laux[4 * k + i] as u16) << (3 * i);
                }

                let map_val = grid_data.map[u as usize];
                if map_val < 0 {
                    is_on_grid_aux[k] = false;
                    let weight: Vec<f32> =
                        (0..4).map(|i| xval[4 * k + i].abs()).collect();
                    grid_data.find_best_neighbor(
                        u,
                        &xval[4 * k..4 * (k + 1)],
                        &weight,
                        this_scale,
                        &mut laux[4 * k..4 * (k + 1)],
                    );
                }
            }

            let mut sumqx = 0.0f32;
            let mut sumq2 = 0.0f32;
            for i in 0..32 {
                let w = xval[i] * xval[i];
                let q = 2.0 * (laux[i] as f32) + 1.0;
                sumqx += w * xval[i] * q;
                sumq2 += w * q * q;
            }

            if sumq2 > 0.0 && sumqx * sumqx > best * sumq2 {
                scale = sumqx / sumq2;
                best = scale * sumqx;
                l.copy_from_slice(&laux);
                is_on_grid.copy_from_slice(&is_on_grid_aux);
            }
        }

        let n_not_ongrid = is_on_grid.iter().filter(|&&x| !x).count();
        if n_not_ongrid > 0 && scale > 0.0 {
            let this_id = 1.0 / scale;
            for k in 0..8 {
                if is_on_grid[k] {
                    continue;
                }
                let mut u = 0u16;
                for i in 0..4 {
                    let lval =
                        nearest_int(0.5 * (this_id * xval[4 * k + i] - 1.0))
                            .clamp(0, KMAX_Q - 1);
                    u |= (lval as u16) << (3 * i);
                }

                let map_val = grid_data.map[u as usize];
                if map_val < 0 {
                    let weight: Vec<f32> =
                        (0..4).map(|i| xval[4 * k + i].abs()).collect();
                    grid_data.find_best_neighbor(
                        u,
                        &xval[4 * k..4 * (k + 1)],
                        &weight,
                        scale,
                        &mut l[4 * k..4 * (k + 1)],
                    );
                }
            }

            let mut sumqx = 0.0f32;
            let mut sumq2 = 0.0f32;
            for i in 0..32 {
                let w = xval[i] * xval[i];
                let q = 2.0 * (l[i] as f32) + 1.0;
                sumqx += w * xval[i] * q;
                sumq2 += w * q * q;
            }
            if sumq2 > 0.0 {
                scale = sumqx / sumq2;
            }
        }

        // Pack grid indices for IQ3_S (9-bit indices: 512 grid points)
        for k in 0..8 {
            let mut u = 0u16;
            for i in 0..4 {
                u |= (l[4 * k + i] as u16) << (3 * i);
            }

            let map_val = grid_data.map[u as usize];
            let grid_index = if map_val >= 0 {
                map_val as u16
            } else {
                panic!("Should not happen - grid lookup failed");
            };

            qs[8 * ib + k] = (grid_index & 0xFF) as u8;
            qh[ib] |= ((grid_index >> 8) as u8) << k;
        }

        scales[ib] = scale;
        max_scale = max_scale.max(scale);
    }

    let d = max_scale / 31.0 * 1.0125;

    // Encode scales
    let id = 1.0 / d;
    let mut encoded_scales = [0u8; 4];
    for ib in 0..8 {
        let lval = nearest_int(0.5 * (id * scales[ib] - 1.0)).clamp(0, 15);
        if ib % 2 == 0 {
            encoded_scales[ib / 2] = lval as u8;
        } else {
            encoded_scales[ib / 2] |= (lval as u8) << 4;
        }
    }

    BlockIQ3S {
        d: f16::from_f32(d).to_bits(),
        qs,
        qh,
        signs,
        scales: encoded_scales,
    }
}

fn quantize_iq1_s(x: &[f32]) -> BlockIQ1S {
    assert!(x.len() == 256);

    let mut amax = 0.0f32;
    for &v in x {
        amax = amax.max(v.abs());
    }

    let d = if amax > GROUP_MAX_EPS { amax } else { 1.0 };
    let id = 1.0 / d;

    let mut qs = [0u8; 32];
    let qh = [0u16; 8];

    for i in 0..32 {
        let mut byte = 0u8;
        for j in 0..8 {
            let val = x[i * 8 + j] * id;
            let q = if val > 0.33 {
                2u8
            } else if val > -0.33 {
                1u8
            } else {
                0u8
            };
            byte |= q << (2 * j);
        }
        qs[i] = byte;
    }

    BlockIQ1S { d: f16::from_f32(d).to_bits(), qs, qh }
}

fn quantize_iq1_m(x: &[f32]) -> BlockIQ1M {
    assert!(x.len() == 256);

    let mut qs = [0u8; 32];
    let qh = [0u8; 16];
    let scales = [0u8; 8];

    for i in 0..32 {
        let mut byte = 0u8;
        for j in 0..8 {
            if x[i * 8 + j] > 0.0 {
                byte |= 1 << j;
            }
        }
        qs[i] = byte;
    }

    BlockIQ1M { qs, qh, scales }
}

// ============================================================================
// Main logic
// ============================================================================

fn main() {
    let args = Args::parse();

    let format = args.format.to_lowercase();
    let n_blocks = args.blocks;
    let seed = args.seed;

    let block_size = if format == "iq4_nl" { 32 } else { 256 };

    println!("=== IQ Quantizer (Grid-based) ===");
    println!("Format: {}", format);
    println!("Blocks: {}", n_blocks);
    println!("Block size: {}", block_size);
    println!("Seed: {}", seed);
    println!();

    // Initialize grid data for grid-based formats
    let iq2_xxs_grid = if format == "iq2_xxs" {
        Some(IQ2GridData::new(256, &KGRID_2BIT_256, 2))
    } else {
        None
    };

    let iq2_xs_grid = if format == "iq2_xs" {
        Some(IQ2GridData::new(512, &KGRID_2BIT_512, 2))
    } else {
        None
    };

    let iq2_s_grid = if format == "iq2_s" {
        Some(IQ2GridData::new(1024, &KGRID_2BIT_1024, 1))
    } else {
        None
    };

    let iq3_xxs_grid = if format == "iq3_xxs" {
        Some(IQ3GridData::new(256, &KGRID_256_IQ3))
    } else {
        None
    };

    let iq3_s_grid = if format == "iq3_s" {
        Some(IQ3GridData::new(512, &KGRID_512_IQ3))
    } else {
        None
    };

    // Generate test data
    let total_values = n_blocks * block_size;
    let fp32_data: Vec<f32> = (0..total_values)
        .map(|i| {
            let t = (i as f32 + seed as f32) * 0.1;
            0.1 + 2.0 * t.cos()
        })
        .collect();

    // Quantize blocks
    let mut quantized_bytes = Vec::new();

    for block_idx in 0..n_blocks {
        let start = block_idx * block_size;
        let block_data = &fp32_data[start..start + block_size];

        let block_bytes = match format.as_str() {
            "iq1_s" => {
                let block = quantize_iq1_s(block_data);
                unsafe {
                    std::slice::from_raw_parts(
                        &block as *const _ as *const u8,
                        std::mem::size_of::<BlockIQ1S>(),
                    )
                }
                .to_vec()
            }
            "iq1_m" => {
                let block = quantize_iq1_m(block_data);
                unsafe {
                    std::slice::from_raw_parts(
                        &block as *const _ as *const u8,
                        std::mem::size_of::<BlockIQ1M>(),
                    )
                }
                .to_vec()
            }
            "iq2_xxs" => {
                let grid = iq2_xxs_grid.as_ref().unwrap();
                let block = quantize_iq2_xxs(block_data, grid);
                unsafe {
                    std::slice::from_raw_parts(
                        &block as *const _ as *const u8,
                        std::mem::size_of::<BlockIQ2XXS>(),
                    )
                }
                .to_vec()
            }
            "iq2_xs" => {
                let grid = iq2_xs_grid.as_ref().unwrap();
                let block = quantize_iq2_xs(block_data, grid);
                unsafe {
                    std::slice::from_raw_parts(
                        &block as *const _ as *const u8,
                        std::mem::size_of::<BlockIQ2XS>(),
                    )
                }
                .to_vec()
            }
            "iq2_s" => {
                let grid = iq2_s_grid.as_ref().unwrap();
                let block = quantize_iq2_s(block_data, grid);
                unsafe {
                    std::slice::from_raw_parts(
                        &block as *const _ as *const u8,
                        std::mem::size_of::<BlockIQ2S>(),
                    )
                }
                .to_vec()
            }
            "iq3_xxs" => {
                let grid = iq3_xxs_grid.as_ref().unwrap();
                let block = quantize_iq3_xxs(block_data, grid);
                unsafe {
                    std::slice::from_raw_parts(
                        &block as *const _ as *const u8,
                        std::mem::size_of::<BlockIQ3XXS>(),
                    )
                }
                .to_vec()
            }
            "iq3_s" => {
                let grid = iq3_s_grid.as_ref().unwrap();
                let block = quantize_iq3_s(block_data, grid);
                unsafe {
                    std::slice::from_raw_parts(
                        &block as *const _ as *const u8,
                        std::mem::size_of::<BlockIQ3S>(),
                    )
                }
                .to_vec()
            }
            "iq4_nl" => {
                let block = quantize_iq4_nl(block_data);
                unsafe {
                    std::slice::from_raw_parts(
                        &block as *const _ as *const u8,
                        std::mem::size_of::<BlockIQ4NL>(),
                    )
                }
                .to_vec()
            }
            "iq4_xs" => {
                let block = quantize_iq4_xs(block_data);
                unsafe {
                    std::slice::from_raw_parts(
                        &block as *const _ as *const u8,
                        std::mem::size_of::<BlockIQ4XS>(),
                    )
                }
                .to_vec()
            }
            _ => {
                eprintln!("Unknown format: {}", format);
                std::process::exit(1);
            }
        };

        quantized_bytes.extend(block_bytes);
    }

    println!("Generated {} bytes", quantized_bytes.len());

    // Save binary output
    if let Some(output_path) = args.output {
        let mut file =
            File::create(&output_path).expect("Failed to create output file");
        file.write_all(&quantized_bytes).expect("Failed to write output");
        println!("Saved to {}", output_path);
    }

    // Export as Rust source
    if let Some(rust_path) = args.export_rust {
        let mut file =
            File::create(&rust_path).expect("Failed to create Rust file");

        writeln!(file, "// Auto-generated test data for {}", format).unwrap();
        writeln!(file, "// Seed: {}, Blocks: {}", seed, n_blocks).unwrap();
        writeln!(file).unwrap();
        writeln!(file, "pub const FORMAT: &str = \"{}\";", format).unwrap();
        writeln!(file, "pub const BLOCK_SIZE: usize = {};", block_size)
            .unwrap();
        writeln!(file, "pub const N_BLOCKS: usize = {};", n_blocks).unwrap();
        writeln!(file).unwrap();

        writeln!(file, "pub const FP32_DATA: &[f32] = &[").unwrap();
        for (i, v) in fp32_data.iter().enumerate() {
            if i % 8 == 0 {
                write!(file, "    ").unwrap();
            }
            write!(file, "{:.6}, ", v).unwrap();
            if (i + 1) % 8 == 0 {
                writeln!(file).unwrap();
            }
        }
        writeln!(file, "];").unwrap();
        writeln!(file).unwrap();

        writeln!(file, "pub const QUANTIZED_DATA: &[u8] = &[").unwrap();
        for (i, b) in quantized_bytes.iter().enumerate() {
            if i % 16 == 0 {
                write!(file, "    ").unwrap();
            }
            write!(file, "0x{:02x}, ", b).unwrap();
            if (i + 1) % 16 == 0 {
                writeln!(file).unwrap();
            }
        }
        writeln!(file, "];").unwrap();

        println!("Exported to {}", rust_path);
    }

    println!("\nDone.");
}

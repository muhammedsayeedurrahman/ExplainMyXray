/* Static demo data matching real API response shapes */
/* 60 orders with fixed coordinates spread across Bengaluru neighborhoods */

export const DEMO_ORDERS = [
  // Koramangala cluster (6 points)
  { id: 1,  customer_name: 'Aarav Sharma',    address: '#42, 5th Main Road, Koramangala 4th Block, Bengaluru',  lat: 12.9352, lng: 77.6245, score: 92, risk: 'green',  explanation: 'Complete address with house number, street, and area.' },
  { id: 2,  customer_name: 'Priya Patel',     address: '#18, 6th Cross, Koramangala 1st Block, Bengaluru',      lat: 12.9341, lng: 77.6198, score: 88, risk: 'green',  explanation: 'Well-structured with cross road and block info.' },
  { id: 3,  customer_name: 'Vikram Singh',    address: 'Near Forum Mall, Koramangala, Bengaluru',               lat: 12.9347, lng: 77.6112, score: 47, risk: 'yellow', explanation: 'Landmark-only address, missing house number and street.' },
  { id: 4,  customer_name: 'Ananya Reddy',    address: '#8, 80 Feet Road, Koramangala 6th Block, Bengaluru',    lat: 12.9380, lng: 77.6260, score: 85, risk: 'green',  explanation: 'Good address with main road and block identifier.' },
  { id: 5,  customer_name: 'Rohan Gupta',     address: '#33, 1st Main, Koramangala 5th Block, Bengaluru',       lat: 12.9325, lng: 77.6280, score: 91, risk: 'green',  explanation: 'Full house number and block reference.' },
  { id: 6,  customer_name: 'Meera Nair',      address: 'Somewhere in Koramangala',                              lat: 12.9360, lng: 77.6155, score: 18, risk: 'red',    explanation: 'Vague area reference, no street or house number.' },

  // HSR Layout cluster (5 points)
  { id: 7,  customer_name: 'Arjun Das',       address: '23, 1st Cross, HSR Layout Sector 1, Bengaluru',         lat: 12.9116, lng: 77.6389, score: 89, risk: 'green',  explanation: 'Well-structured with cross road and sector info.' },
  { id: 8,  customer_name: 'Kavya Iyer',      address: '#45, 14th Main, HSR Layout Sector 4, Bengaluru',        lat: 12.9082, lng: 77.6412, score: 86, risk: 'green',  explanation: 'Complete with main road and sector reference.' },
  { id: 9,  customer_name: 'Siddharth Rao',   address: 'Near BDA Complex, HSR Layout, Bengaluru',               lat: 12.9140, lng: 77.6350, score: 52, risk: 'yellow', explanation: 'Landmark-only, missing house number.' },
  { id: 10, customer_name: 'Neha Joshi',      address: '#12, 27th Main, HSR Layout Sector 2, Bengaluru',        lat: 12.9098, lng: 77.6445, score: 90, risk: 'green',  explanation: 'Full address with main road and sector.' },
  { id: 11, customer_name: 'Aditya Kumar',    address: 'HSR Layout',                                            lat: 12.9130, lng: 77.6320, score: 12, risk: 'red',    explanation: 'Area name only — extremely vague.' },

  // Indiranagar cluster (5 points)
  { id: 12, customer_name: 'Divya Menon',     address: '#8, 10th Main, Indiranagar, Bengaluru',                  lat: 12.9784, lng: 77.6408, score: 87, risk: 'green',  explanation: 'Good address with main road and area identifier.' },
  { id: 13, customer_name: 'Karthik Hegde',   address: '#22, 100 Feet Road, Indiranagar, Bengaluru',            lat: 12.9810, lng: 77.6380, score: 84, risk: 'green',  explanation: 'House number with major road reference.' },
  { id: 14, customer_name: 'Riya Bhat',       address: 'Near Toit Brewery, Indiranagar, Bengaluru',             lat: 12.9795, lng: 77.6440, score: 55, risk: 'yellow', explanation: 'Landmark reference only, no house number.' },
  { id: 15, customer_name: 'Suresh Gowda',    address: '#5, 12th Cross, Indiranagar 2nd Stage, Bengaluru',      lat: 12.9770, lng: 77.6350, score: 91, risk: 'green',  explanation: 'Complete with cross road and stage reference.' },
  { id: 16, customer_name: 'Lakshmi Iyengar', address: 'Behind some shop, Indiranagar',                         lat: 12.9820, lng: 77.6460, score: 15, risk: 'red',    explanation: 'Vague landmark, no verifiable location.' },

  // Whitefield cluster (5 points)
  { id: 17, customer_name: 'Rajesh Murthy',   address: 'Flat 301, Prestige Shantiniketan, Whitefield, Bengaluru', lat: 12.9698, lng: 77.7510, score: 93, risk: 'green', explanation: 'Full flat number, building name, and locality.' },
  { id: 18, customer_name: 'Deepa Shetty',    address: '#7, ITPL Main Road, Whitefield, Bengaluru',              lat: 12.9720, lng: 77.7480, score: 82, risk: 'green', explanation: 'House number with major road reference.' },
  { id: 19, customer_name: 'Mohan Kulkarni',  address: 'Near Phoenix Mall, Whitefield',                          lat: 12.9680, lng: 77.7540, score: 48, risk: 'yellow', explanation: 'Landmark reference, missing street details.' },
  { id: 20, customer_name: 'Sneha Acharya',   address: '#102, Prestige Tech Park, Whitefield, Bengaluru',        lat: 12.9740, lng: 77.7420, score: 88, risk: 'green', explanation: 'Complete with building and area reference.' },
  { id: 21, customer_name: 'Amit Verma',      address: 'Whitefield',                                             lat: 12.9660, lng: 77.7560, score: 10, risk: 'red',   explanation: 'Area name only — extremely vague.' },

  // Rajajinagar cluster (5 points)
  { id: 22, customer_name: 'Pooja Rao',       address: 'Near ISKCON Temple, Rajajinagar, Bengaluru',             lat: 12.9895, lng: 77.5530, score: 50, risk: 'yellow', explanation: 'Landmark-only address, missing house number and street.' },
  { id: 23, customer_name: 'Ganesh Bhat',     address: '#15, 4th Block, Rajajinagar, Bengaluru',                 lat: 12.9910, lng: 77.5495, score: 83, risk: 'green',  explanation: 'House number with block and area.' },
  { id: 24, customer_name: 'Nandini Hegde',   address: 'Opp Orion Mall, Rajajinagar, Bengaluru',                 lat: 12.9880, lng: 77.5560, score: 56, risk: 'yellow', explanation: 'Landmark reference only, no house number.' },
  { id: 25, customer_name: 'Ramesh Iyengar',  address: '#44, Chord Road, Rajajinagar, Bengaluru',                lat: 12.9920, lng: 77.5470, score: 87, risk: 'green',  explanation: 'Complete with road name and area.' },
  { id: 26, customer_name: 'Shilpa Naik',     address: 'Near bus stop, Rajajinagar',                             lat: 12.9870, lng: 77.5585, score: 20, risk: 'red',    explanation: 'Vague landmark, no verifiable location.' },

  // Malleshwaram cluster (4 points)
  { id: 27, customer_name: 'Venkat Reddy',    address: 'Opp Mantri Mall, Malleshwaram, Bengaluru',               lat: 12.9960, lng: 77.5710, score: 58, risk: 'yellow', explanation: 'Landmark reference only, no house number.' },
  { id: 28, customer_name: 'Asha Prasad',     address: '#23, 8th Cross, Malleshwaram, Bengaluru',                lat: 12.9945, lng: 77.5680, score: 90, risk: 'green',  explanation: 'Complete address with house number and cross road.' },
  { id: 29, customer_name: 'Chetan Gowda',    address: '#7, Sampige Road, Malleshwaram, Bengaluru',              lat: 12.9975, lng: 77.5735, score: 85, risk: 'green',  explanation: 'House number with named road.' },
  { id: 30, customer_name: 'Parvathi Nair',   address: 'Malleshwaram area',                                      lat: 12.9935, lng: 77.5750, score: 14, risk: 'red',    explanation: 'Area name only — extremely vague.' },

  // JP Nagar cluster (4 points)
  { id: 31, customer_name: 'Manoj Kumar',     address: '#5, 12th Main, JP Nagar 2nd Phase, Bengaluru',           lat: 12.9050, lng: 77.5850, score: 84, risk: 'green',  explanation: 'House number with main road and phase.' },
  { id: 32, customer_name: 'Sarita Bhat',     address: '#19, 15th Cross, JP Nagar 5th Phase, Bengaluru',         lat: 12.9020, lng: 77.5880, score: 86, risk: 'green',  explanation: 'Complete with cross and phase reference.' },
  { id: 33, customer_name: 'Prasad Gowda',    address: 'Near Raghu Theatre, JP Nagar, Bengaluru',                lat: 12.9070, lng: 77.5820, score: 51, risk: 'yellow', explanation: 'Landmark-only, vague locality reference.' },
  { id: 34, customer_name: 'Latha Devi',      address: 'JP Nagar somewhere',                                     lat: 12.9035, lng: 77.5900, score: 11, risk: 'red',    explanation: 'Vague area, no verifiable address.' },

  // Basavanagudi cluster (4 points)
  { id: 35, customer_name: 'Sundar Krishnan', address: '#3, Bull Temple Road, Basavanagudi, Bengaluru',          lat: 12.9430, lng: 77.5680, score: 88, risk: 'green',  explanation: 'Complete with road name and area.' },
  { id: 36, customer_name: 'Geeta Raman',     address: '#11, Gandhi Bazaar, Basavanagudi, Bengaluru',            lat: 12.9445, lng: 77.5720, score: 85, risk: 'green',  explanation: 'House number with landmark road.' },
  { id: 37, customer_name: 'Balu Reddy',      address: 'Near Lalbagh Gate, Basavanagudi, Bengaluru',             lat: 12.9460, lng: 77.5750, score: 60, risk: 'yellow', explanation: 'Landmark-only, missing house number.' },
  { id: 38, customer_name: 'Uma Shankar',     address: 'Behind temple, Basavanagudi',                            lat: 12.9415, lng: 77.5650, score: 16, risk: 'red',    explanation: 'Incomplete — only relative direction.' },

  // MG Road / Central (4 points)
  { id: 39, customer_name: 'Nikhil Sharma',   address: '12, Residency Road, Shanthala Nagar, Bengaluru',         lat: 12.9716, lng: 77.6050, score: 90, risk: 'green',  explanation: 'Well-structured commercial area address.' },
  { id: 40, customer_name: 'Rekha Iyengar',   address: 'Brigade Road, Ashok Nagar, Bengaluru',                   lat: 12.9730, lng: 77.6080, score: 62, risk: 'yellow', explanation: 'Road name only, needs more specifics.' },
  { id: 41, customer_name: 'Harish Murthy',   address: '#19, Cunningham Road, Vasanth Nagar, Bengaluru',         lat: 12.9880, lng: 77.5920, score: 89, risk: 'green',  explanation: 'Good structure with road and area.' },
  { id: 42, customer_name: 'Sudha Verma',     address: 'Near Vidhana Soudha, Ambedkar Veedhi, Bengaluru',        lat: 12.9790, lng: 77.5910, score: 81, risk: 'green',  explanation: 'Landmark reference near government building.' },

  // Marathahalli / ORR (5 points)
  { id: 43, customer_name: 'Vinay Prasad',    address: 'Outer Ring Road, Marathahalli, Bengaluru',               lat: 12.9560, lng: 77.7010, score: 58, risk: 'yellow', explanation: 'Major road reference, apartment number missing.' },
  { id: 44, customer_name: 'Swathi Bhat',     address: '#203, Adarsh Palm Retreat, Bellandur, Bengaluru',        lat: 12.9260, lng: 77.6780, score: 94, risk: 'green',  explanation: 'Full flat number, building name, and locality.' },
  { id: 45, customer_name: 'Raghu Nandan',    address: '#8, Sarjapur Road, Bellandur, Bengaluru',                lat: 12.9240, lng: 77.6820, score: 83, risk: 'green',  explanation: 'House number with major road.' },
  { id: 46, customer_name: 'Padma Rao',       address: 'Near Innovative Multiplex, Marathahalli',                lat: 12.9580, lng: 77.6980, score: 49, risk: 'yellow', explanation: 'Landmark reference, no street details.' },
  { id: 47, customer_name: 'Sunil Gowda',     address: 'Bluru 560037',                                           lat: 12.9540, lng: 77.7050, score: 13, risk: 'red',    explanation: 'Misspelled city, PIN only — very low quality.' },

  // Electronic City (4 points)
  { id: 48, customer_name: 'Nisha Kumari',    address: '#55, Phase 1, Electronic City, Bengaluru',               lat: 12.8440, lng: 77.6720, score: 86, risk: 'green',  explanation: 'House number with phase and area.' },
  { id: 49, customer_name: 'Bharath Raj',     address: '#120, Infosys Campus Road, Electronic City, Bengaluru',  lat: 12.8410, lng: 77.6690, score: 91, risk: 'green',  explanation: 'Complete with road and area reference.' },
  { id: 50, customer_name: 'Yamuna Devi',     address: 'Electronic City',                                        lat: 12.8460, lng: 77.6750, score: 15, risk: 'red',    explanation: 'Area name only — extremely vague.' },
  { id: 51, customer_name: 'Shiva Kumar',     address: 'Near Wipro Gate, Electronic City Phase 2, Bengaluru',    lat: 12.8385, lng: 77.6660, score: 54, risk: 'yellow', explanation: 'Landmark with phase but no house number.' },

  // RR Nagar / West (3 points)
  { id: 52, customer_name: 'Pallavi Hegde',   address: '#67, 80 Feet Road, RR Nagar, Bengaluru',                 lat: 12.9580, lng: 77.5230, score: 83, risk: 'green',  explanation: 'Has house number and road, good deliverability.' },
  { id: 53, customer_name: 'Dinesh Shetty',   address: '#12, BEML Layout, RR Nagar, Bengaluru',                  lat: 12.9560, lng: 77.5270, score: 85, risk: 'green',  explanation: 'House number with layout reference.' },
  { id: 54, customer_name: 'Jayanti Devi',    address: 'Some place near bus stop, RR Nagar',                     lat: 12.9600, lng: 77.5200, score: 19, risk: 'red',    explanation: 'Vague landmark, no verifiable location.' },

  // Yelahanka / North (3 points)
  { id: 55, customer_name: 'Prabhu Rajan',    address: '#9, Kogilu Cross, Yelahanka, Bengaluru',                 lat: 13.0980, lng: 77.5940, score: 82, risk: 'green',  explanation: 'House number with cross road.' },
  { id: 56, customer_name: 'Keerthi Nair',    address: '#34, New Town, Yelahanka, Bengaluru',                    lat: 13.1010, lng: 77.5910, score: 80, risk: 'green',  explanation: 'House number with area reference.' },
  { id: 57, customer_name: 'Raju Patil',      address: 'Near Air Force Station, Yelahanka',                      lat: 13.0950, lng: 77.5970, score: 45, risk: 'yellow', explanation: 'Landmark reference, missing street.' },

  // Jayanagar (3 points)
  { id: 58, customer_name: 'Savitha Rao',     address: '#22, 11th Main, Jayanagar 4th Block, Bengaluru',         lat: 12.9250, lng: 77.5830, score: 92, risk: 'green',  explanation: 'Complete with house number, main road, and block.' },
  { id: 59, customer_name: 'Manohar Das',     address: '#6, 30th Cross, Jayanagar 9th Block, Bengaluru',         lat: 12.9220, lng: 77.5810, score: 87, risk: 'green',  explanation: 'Full address with cross road and block.' },
  { id: 60, customer_name: 'Hema Srinivasan', address: 'Jayanagar Complex',                                      lat: 12.9270, lng: 77.5860, score: 22, risk: 'red',    explanation: 'Vague area name, no house number or street.' },
]

export const DEMO_METRICS = {
  total_orders: 400,
  average_score: 68,
  deliveries_at_risk: 72,
  trips_before: 84,
  trips_after: 31,
  trips_saved: 53,
  cost_saved: 6360,
  failure_rate_before: 18.0,
  failure_rate_after: 5.2,
  clusters_formed: 14,
  verified_addresses: 312,
  trip_efficiency: 63,
  address_verification: 78,
  failure_reduction: 71,
  score_distribution: { green: 248, yellow: 80, red: 72 },
  industry_benchmarks: {
    last_mile_cost_pct: 53,
    rto_rate_tier2_3: 25,
    daily_ecommerce_deliveries: '10M+',
    logistics_gdp_pct: 14,
    global_logistics_gdp_pct: 8,
  },
}

export const DEMO_INSIGHTS = [
  {
    type: 'failure_pattern',
    icon: 'alert',
    title: '38% of failures from landmark-only addresses',
    text: 'Addresses like "Near ISKCON Temple" lack house numbers and streets, causing delivery agents to spend 15+ min locating recipients.',
  },
  {
    type: 'risk_concentration',
    icon: 'warning',
    title: 'Rajajinagar has highest risk density',
    text: '14 of 72 at-risk deliveries are concentrated in Rajajinagar — older area with inconsistent numbering systems.',
  },
  {
    type: 'area_analysis',
    icon: 'map',
    title: 'Koramangala leads in address quality',
    text: 'Average score 87/100 — newer planned layouts with consistent block numbering and well-mapped streets.',
  },
  {
    type: 'optimization',
    icon: 'sparkle',
    title: 'PackBuddy reduced trips by 63%',
    text: 'DBSCAN clustering grouped 400 orders into 14 delivery zones. Trip count dropped from 84 to 31.',
  },
  {
    type: 'industry',
    icon: 'chart',
    title: 'India loses $4.6B annually to failed deliveries',
    text: 'Address quality is the #1 cause. SmartRoute\'s approach could reduce RTO rates by up to 70% in Tier 2/3 cities.',
  },
]

export const DEMO_SCORE_RESULT = {
  score: 47,
  risk: 'yellow',
  explanation: 'Landmark-based address — missing house number and street name. PIN code not provided. Area "Rajajinagar" recognized but insufficient for precise delivery.',
}

export const DEMO_VERIFY_RESULT = {
  score: 81,
  risk: 'green',
  explanation: 'AI-corrected address with inferred house number, validated street, and confirmed PIN code.',
  corrected_address: '#12, 3rd Cross, Rajajinagar 1st Block, Bengaluru',
  corrected_pin: '560010',
  corrections: [
    'Added inferred house number "#12" from nearest known delivery point',
    'Resolved "Near ISKCON Temple" to "3rd Cross, Rajajinagar 1st Block"',
    'Added missing PIN code 560010 for Rajajinagar',
  ],
}

import { EmergencyContact, StateLegalService, EmergencyGuide } from '@/types';

export const nationalHelplines: EmergencyContact[] = [
  {
    name: 'Police Emergency',
    number: '112',
    description: 'Universal emergency number for police, fire, and ambulance',
    available24x7: true,
    tollFree: true,
  },
  {
    name: 'Women Helpline',
    number: '181',
    description: 'National Commission for Women helpline for women in distress and domestic violence cases',
    available24x7: true,
    tollFree: true,
  },
  {
    name: 'Child Helpline',
    number: '1098',
    description: 'CHILDLINE India for children in need of care and protection',
    available24x7: true,
    tollFree: true,
  },
  {
    name: 'Cyber Crime Helpline',
    number: '1930',
    description: 'National Cyber Crime Reporting helpline',
    available24x7: true,
    tollFree: true,
  },
  {
    name: 'National Legal Services',
    number: '15100',
    description: 'NALSA helpline for free legal aid and advice',
    available24x7: false,
    tollFree: true,
  },
  {
    name: 'Consumer Helpline',
    number: '1800-11-4000',
    description: 'National Consumer Helpline for consumer complaints',
    available24x7: false,
    tollFree: true,
  },
  {
    name: 'Senior Citizen Helpline',
    number: '14567',
    description: 'Elder Line for senior citizens in distress',
    available24x7: true,
    tollFree: true,
  },
];

export const stateLegalServices: StateLegalService[] = [
  {
    state: 'Delhi',
    authority: 'Delhi State Legal Services Authority',
    phone: '011-23385023',
    address: 'Patiala House Courts, New Delhi - 110001',
    website: 'https://dslsa.org',
  },
  {
    state: 'Maharashtra',
    authority: 'Maharashtra State Legal Services Authority',
    phone: '022-22027273',
    address: 'High Court Building, Mumbai - 400032',
    website: 'https://maharashtralsa.gov.in',
  },
  {
    state: 'Tamil Nadu',
    authority: 'Tamil Nadu State Legal Services Authority',
    phone: '044-25361923',
    address: 'High Court Buildings, Chennai - 600104',
    website: 'https://tnslsa.gov.in',
  },
  {
    state: 'Karnataka',
    authority: 'Karnataka State Legal Services Authority',
    phone: '080-22110698',
    address: 'Nrupathunga Road, Bangalore - 560001',
    website: 'https://kslsa.kar.nic.in',
  },
  {
    state: 'West Bengal',
    authority: 'West Bengal State Legal Services Authority',
    phone: '033-22435933',
    address: 'High Court Building, Kolkata - 700001',
    website: 'https://wbslsa.gov.in',
  },
  {
    state: 'Uttar Pradesh',
    authority: 'UP State Legal Services Authority',
    phone: '0522-2209116',
    address: 'High Court, Lucknow - 226001',
    website: 'https://upslsa.gov.in',
  },
  {
    state: 'Rajasthan',
    authority: 'Rajasthan State Legal Services Authority',
    phone: '0141-2227481',
    address: 'High Court Premises, Jodhpur - 342001',
    website: 'https://rlsa.gov.in',
  },
  {
    state: 'Gujarat',
    authority: 'Gujarat State Legal Services Authority',
    phone: '079-27683771',
    address: 'High Court Building, Ahmedabad - 380009',
    website: 'https://gujslsa.gov.in',
  },
  {
    state: 'Kerala',
    authority: 'Kerala State Legal Services Authority',
    phone: '0471-2579911',
    address: 'High Court of Kerala, Kochi - 682031',
    website: 'https://kelsa.nic.in',
  },
  {
    state: 'Telangana',
    authority: 'Telangana State Legal Services Authority',
    phone: '040-23234343',
    address: 'High Court Building, Hyderabad - 500066',
    website: 'https://tslsa.telangana.gov.in',
  },
  {
    state: 'Andhra Pradesh',
    authority: 'AP State Legal Services Authority',
    phone: '0866-2432668',
    address: 'High Court of AP, Amaravati',
    website: 'https://apslsa.ap.gov.in',
  },
  {
    state: 'Punjab',
    authority: 'Punjab State Legal Services Authority',
    phone: '0172-2744001',
    address: 'Punjab and Haryana High Court, Chandigarh',
    website: 'https://pulsa.gov.in',
  },
  {
    state: 'Bihar',
    authority: 'Bihar State Legal Services Authority',
    phone: '0612-2223832',
    address: 'Patna High Court, Patna - 800001',
    website: 'https://bslsa.bih.nic.in',
  },
  {
    state: 'Madhya Pradesh',
    authority: 'MP State Legal Services Authority',
    phone: '0755-2577523',
    address: 'High Court Premises, Jabalpur - 482001',
    website: 'https://mpslsa.gov.in',
  },
  {
    state: 'Odisha',
    authority: 'Odisha State Legal Services Authority',
    phone: '0674-2392092',
    address: 'High Court Campus, Cuttack - 753002',
    website: 'https://oslsa.nic.in',
  },
];

export const emergencyGuides: EmergencyGuide[] = [
  {
    id: 'arrested',
    title: 'If You Are Arrested',
    description: 'Know your rights during arrest and detention',
    steps: [
      {
        title: 'Stay calm and cooperate',
        description: 'Do not resist arrest. Stay calm and polite. Resisting can lead to additional charges.',
      },
      {
        title: 'Ask for the reason',
        description: 'You have the right to know why you are being arrested. Ask the officer to state the reason clearly.',
      },
      {
        title: 'Inform family or lawyer',
        description: 'Under Article 22(1), you have the right to inform a family member or friend. Ask to make a phone call immediately.',
      },
      {
        title: 'Do not sign anything',
        description: 'Do not sign any blank paper or document you haven\'t read. You have the right to read everything before signing.',
      },
      {
        title: 'Request legal aid',
        description: 'If you cannot afford a lawyer, you have the right to free legal aid under Article 39A. Ask for a legal aid lawyer.',
      },
      {
        title: 'Must be produced in court within 24 hours',
        description: 'Under Article 22(2), police must present you before a magistrate within 24 hours of arrest.',
      },
    ],
  },
  {
    id: 'domestic-violence',
    title: 'If You Face Domestic Violence',
    description: 'Immediate steps to take if you or someone you know faces domestic violence',
    steps: [
      {
        title: 'Ensure immediate safety',
        description: 'If in danger, call 112 (police) or 181 (women helpline) immediately. Move to a safe location if possible.',
      },
      {
        title: 'Document injuries',
        description: 'If injured, get medical help. Ask the doctor to document injuries in a medico-legal case (MLC) report.',
      },
      {
        title: 'File a complaint',
        description: 'File a Domestic Incident Report (DIR) with the Protection Officer or police station. A complaint under the DV Act is free.',
      },
      {
        title: 'Seek protection order',
        description: 'Apply for a protection order from the Magistrate under Section 18 of the Protection of Women from Domestic Violence Act.',
      },
      {
        title: 'Contact support organizations',
        description: 'Reach out to local NGOs, One Stop Centres, or women\'s shelters for support and guidance.',
      },
    ],
  },
  {
    id: 'cyber-fraud',
    title: 'If You Are a Victim of Online Fraud',
    description: 'Steps to take immediately after discovering online fraud',
    steps: [
      {
        title: 'Block the transaction',
        description: 'Call your bank immediately to block your card/account. Most banks have 24x7 helplines. Time is critical.',
      },
      {
        title: 'Report to Cyber Crime portal',
        description: 'File a complaint at cybercrime.gov.in or call 1930. Report within the golden hour for best chance of recovery.',
      },
      {
        title: 'File an FIR',
        description: 'Visit the nearest police station or cyber crime cell and file an FIR with all evidence.',
      },
      {
        title: 'Preserve evidence',
        description: 'Take screenshots of transactions, messages, emails, and websites. Do not delete any communication.',
      },
      {
        title: 'Report to RBI if banking fraud',
        description: 'If unauthorized transaction, report to RBI. Under RBI guidelines, liability is limited if reported within 3 days.',
      },
    ],
  },
];

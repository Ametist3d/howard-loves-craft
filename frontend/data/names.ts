import { Language } from '../types';

export const FIRST_NAMES = [
  'Simon', 'Robert', 'Francis', 'Peter', 'Aaron', 'Ida', 'Gideon', 'Iris', 'Edward', 'Marilyn', 'Oliphant',
  'Eliza', 'Emma', 'Isabella', 'Owen', 'Noah', 'Grace', 'Rufus', 'Seraphina', 'Silas', 'Virgil', 'Julian',
  'Margaret', 'Garret', 'Patrick', 'Arthur', 'William', 'Henry', 'Thomas', 'John', 'Charles', 'Theodore',
  'Harvey', 'Eleanor', 'Mary', 'Alice', 'Edith', 'Beatrice', 'Martha', 'Dorothy', 'Lillian', 'Agatha',
];

export const LAST_NAMES = [
  'Croft', 'Ivers', 'Burton', 'Darley', 'Bledsoe', 'Mortimer', 'Keith', 'Wells', 'Hall', 'Shaw', 'Crow',
  'Thorne', 'Kingston', 'Frey', 'Doyle', 'Carmichael', 'Elder', 'Ray', 'Chase', 'Ward', 'Stayton', 'Moore',
  'Lancaster', 'Lang', 'Morell', 'Blackwood', 'Kowalski', 'Smith', 'Vance', 'Malone', 'Armitage', 'Lovecraft',
  'Carter', 'Marsh', 'Gilman', 'Pickman', 'Phillips', 'West',
];

export const getRandomName = (_language: Language): string => {
  const first = FIRST_NAMES[Math.floor(Math.random() * FIRST_NAMES.length)];
  const last = LAST_NAMES[Math.floor(Math.random() * LAST_NAMES.length)];
  return `${first} ${last}`;
};
# Create a new file named `doubly_linked_list.py` in the same directory as your script.
write_file('doubly_linked_list.py', '''
class Node:
    def __init__(self, data):
        self.data = data
        self.prev = None
        self.next = None

class DoublyLinkedList:
    def __init__(self):
        self.head = Node(None)
        self.tail = Node(None)
        self.head.prev = self.tail
        self.tail.next = self.head

    def append(self, data):
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            self.tail = new_node
        else:
            self.tail.next = new_node
            new_node.prev = self.tail
            self.tail = new_node

    def prepend(self, data):
        new_node = Node(data)
        if not self.tail:
            self.head = new_node
            self.tail = new_node
        else:
            self.head.prev = new_node
            new_node.next = self.head
            self.head = new_node

    def remove(self, node):
        if node == self.head:
            self.head = node.next
            if node.next:
                node.next.prev = None
        elif node == self.tail:
            self.tail = node.prev
            if node.prev:
                node.prev.next = None
        else:
            node.prev.next = node.next
            node.next.prev = node.prev

    def traverse(self):
        current = self.head
        while current != self.tail:
            print(current.data, end=" ")
            current = current.next
        print(current.data)  # Print the last element as well
''')

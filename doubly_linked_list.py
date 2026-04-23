import os

def write_file(path, content):
    with open(path, 'w') as file:
        file.write(content)

# Node class to represent each element in the doubly linked list
class Node:
    def __init__(self, data):
        self.data = data
        self.prev = None
        self.next = None

# DoublyLinkedList class to manage the nodes
class DoublyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None

    # Method to add a new node at the head of the list
    def add_to_head(self, value):
        new_node = Node(value)
        if self.head is None:
            self.head = self.tail = new_node
        else:
            new_node.next = self.head
            self.head.prev = new_node
            self.head = new_node

    # Method to add a new node at the tail of the list
    def add_to_tail(self, value):
        new_node = Node(value)
        if self.tail is None:
            self.head = self.tail = new_node
        else:
            new_node.prev = self.tail
            self.tail.next = new_node
            self.tail = new_node

    # Method to remove a specified node from the list
    def remove_node(self, node):
        if node.prev:
            node.prev.next = node.next
        if node.next:
            node.next.prev = node.prev
        if node == self.head:
            self.head = node.next
        if node == self.tail:
            self.tail = node.prev

    # Method to search for a value in the list and return the corresponding node
    def search(self, value):
        current_node = self.head
        while current_node is not None:
            if current_node.data == value:
                return current_node
            current_node = current_node.next
        return None

# Example usage
if __name__ == "__main__":
    dll = DoublyLinkedList()
    
    # Add nodes to the list
    dll.add_to_head(1)
    dll.add_to_tail(2)
    dll.add_to_tail(3)
    
    # Search for a value
    node = dll.search(2)
    if node:
        print(f"Node with value {node.data} found.")
    else:
        print("Node not found.")

    # Remove the node
    dll.remove_node(node)

# Write the corrected files to the specified path
write_file('doubly_linked_list.py', open('doubly_linked_list.py').read())

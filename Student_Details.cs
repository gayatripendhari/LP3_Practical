//SPDX-License-Identifier: MIT
pragma solidity ^0.8.7;
contract Student_management{
    // Structure Defined for Student Details
    struct Student{
        int stud_id;
        string Name;
        string department;
    }

    Student[] Students; // Students Array Declaration

    // Function to add Students
    function add_stud(int stud_id, string memory Name, string memory department) public{
        Student memory stud = Student(stud_id, Name, department);
        Students.push(stud);
    }

    // Function to display Student Details
    function getStudent(int stud_id) public view returns(string memory, string memory){
        for(uint i=0; i< Students.length; i++){
            Student memory stud = Students[i];
            if(stud.stud_id == stud_id){
                return(stud.Name, stud.department);
            }
        }
        return("Student Information not found...!", "Not Found"); 
    }
}

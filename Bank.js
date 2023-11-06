// SPDX-License-Identifier: MIT
pragma solidity 0.8.7;
contract Bank{
    address public owner;
    constructor() {
        owner=msg.sender;
    }
    //function to deposit money
    function deposit() external payable{
        require(msg.value ==2 ether,"Please send Two Ether");
    }

    // Function to send money
    function send(address payable _to)external payable OnlyBy() {
        require(msg.sender==owner, "No");
        _to.transfer(1000000000000000000);
    }
   
    // Modifier for send function which allows only owner to send money
    modifier OnlyBy(){
        if (msg.sender==owner){
        _;
        }
        
    }
}